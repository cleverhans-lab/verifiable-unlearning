use circ::front::zsharp::{self, ZSharpFE};
use circ::front::{FrontEnd, Mode};
use circ::ir::{
    opt::{opt, Opt},
    term::text::parse_value_map,
};

use circ::target::r1cs::opt::reduce_linearities;
use circ::target::r1cs::spartan::int_to_scalar;
use circ::target::r1cs::trans::to_r1cs;
use circ::target::r1cs::spartan;

use circ::util::field::DFL_T;
use circ_fields::FieldT;

use std::path::PathBuf;
use structopt::StructOpt;
use structopt::clap::arg_enum;

use std::time::Instant;
use libspartan::{InputsAssignment, NIZKGens, NIZK};
use merlin::Transcript;

use libspartan::{SNARKGens, SNARK};

#[derive(Debug, StructOpt)]
#[structopt(name = "circ", about = "CirC: the circuit compiler")]
struct Options {
    /// Input file
    #[structopt(parse(from_os_str), name = "PATH")]
    path: PathBuf,

    /// Proof variant
    #[structopt(possible_values=&ProofVariant::variants(), case_insensitive=true)]
    variant: ProofVariant,    

    /// Prover inputs
    #[structopt(long, default_value = "pin", parse(from_os_str))]
    pin: PathBuf,
    /// Verifier inputs
    #[structopt(long, default_value = "vin", parse(from_os_str))]
    vin: PathBuf,


    #[structopt(long, default_value = "50")]
    /// linear combination constraints up to this size will be eliminated
    lc_elimination_thresh: usize,

    /// In Z#, "isolate" assertions. That is, assertions in if/then/else expressions only take
    /// effect if that branch is active.
    ///
    /// See `--branch-isolation` in
    /// [ZoKrates](https://zokrates.github.io/language/control_flow.html).
    #[structopt(long)]
    z_isolate_asserts: bool,

}

arg_enum! {
    #[derive(PartialEq, Eq, Debug)]
    enum ProofVariant {
        SNARK,
        NIZK
    }
}

fn main() {
    env_logger::Builder::from_default_env()
        .format_level(false)
        .format_timestamp(None)
        .init();
    let options = Options::from_args();

    match options.variant {

        ProofVariant::NIZK => {

            /**************************
            ******     SETUP     ******
            ***************************/

            let setup_timer = Instant::now();
            println!("\n[+] Compile circuit");
            let timer = Instant::now();
            let mode = Mode::Proof;
            let inputs = zsharp::Inputs {
                file: options.path,
                mode,
                isolate_asserts: options.z_isolate_asserts,
            };
            let cs =  ZSharpFE::gen(inputs);
            println!("    took {} ms", timer.elapsed().as_millis());

            println!("\n[+] Optimize circuit");
            let timer = Instant::now();
            let cs = opt(
                cs,
                vec![
                    Opt::ScalarizeVars,
                    Opt::Flatten,
                    Opt::Sha,
                    Opt::ConstantFold(Box::new([])),
                    Opt::Flatten,
                    Opt::Inline,
                    // Tuples must be eliminated before oblivious array elim
                    Opt::Tuple,
                    Opt::ConstantFold(Box::new([])),
                    Opt::Obliv,
                    // The obliv elim pass produces more tuples, that must be eliminated
                    Opt::Tuple,
                    Opt::LinearScan,
                    // The linear scan pass produces more tuples, that must be eliminated
                    Opt::Tuple,
                    Opt::Flatten,
                    Opt::ConstantFold(Box::new([])),
                    Opt::Inline,
                ]
            );
            println!("    took {} ms", timer.elapsed().as_millis());

            println!("\n[+] Converting to R1CS");
            let timer = Instant::now();
            let (r1cs, mut prover_data, verifier_data) =
                to_r1cs(cs.get("main").clone(), FieldT::from(DFL_T.modulus()));
            println!("    - pre-opt R1CS size: {}", r1cs.constraints().len());
            let r1cs = reduce_linearities(r1cs, Some(options.lc_elimination_thresh));
            println!("    - final R1CS size: {}", r1cs.constraints().len());
            // save the optimized r1cs: the prover needs it to synthesize.
            prover_data.r1cs = r1cs;
            println!("    took {} ms", timer.elapsed().as_millis());

            println!("\n[+] Get instance size");
            let timer = Instant::now();
            let inputs = parse_value_map(&std::fs::read(PathBuf::from(options.pin.as_path())).unwrap());
            let (_inst, _wit, _inps, num_cons, num_vars, num_inputs, num_non_zero_entries) =
                spartan::r1cs_to_spartan(&prover_data, &inputs);
            assert_ne!(num_cons, 0, "No constraints");
            println!("    took {} ms", timer.elapsed().as_millis());
            println!("    num_cons {}", num_cons);
            println!("    num_vars {}", num_vars);
            println!("    num_inputs {}", num_inputs);
            println!("    num_non_zero_entries {}", num_non_zero_entries);

            println!("\n[+] Generate public parameters");
            let timer = Instant::now();
            let gens = NIZKGens::new(num_cons, num_vars, num_inputs);
            println!("    took {} ms", timer.elapsed().as_millis());

            println!("\n[+] Setup total {} ms", setup_timer.elapsed().as_millis());

            /**************************
            ******    PROVING    ******
            ***************************/

            println!("\n[+] Prove");
            let timer = Instant::now();
        
            let prover_input_map = parse_value_map(&std::fs::read(options.pin).unwrap());
            let (inst, wit, inps, _num_cons, _num_vars, _num_inputs, _num_non_zero_entries) =
                spartan::r1cs_to_spartan(&prover_data, &prover_input_map);
        
            let mut prover_transcript = Transcript::new(b"nizk_example");
            let proof = NIZK::prove(&inst, wit, &inps, &gens, &mut prover_transcript);
            println!("    took {} ms", timer.elapsed().as_millis());
            
            // proof size
            let innerproof = &proof.r1cs_sat_proof;
            let proof_len = bincode::serialize(innerproof).unwrap().len();
            println!("    proof size {}", proof_len);

            /**************************
            ******   VERIYIYNG   ******
            ***************************/

            println!("\n[+] Verify");
            let timer = Instant::now();
            let verifier_input_map = parse_value_map(&std::fs::read(options.vin).unwrap());
            let values = verifier_data.eval(&verifier_input_map);
            
            let mut inp = Vec::new();
            for v in &values {
                let scalar = int_to_scalar(v);
                inp.push(scalar.to_bytes());
            }
            let inputs = InputsAssignment::new(&inp).unwrap();
            let mut verifier_transcript = Transcript::new(b"nizk_example");
            assert!(proof
                .verify(&inst, &inputs, &mut verifier_transcript, &gens)
                .is_ok());
            println!("    took {} ms", timer.elapsed().as_millis());
        
        }

        ProofVariant::SNARK => {

            /**************************
            ******     SETUP     ******
            ***************************/

            let setup_timer = Instant::now();
            println!("\n[+] Compile circuit");
            let timer = Instant::now();
            let mode = Mode::Proof;
            let inputs = zsharp::Inputs {
                file: options.path,
                mode,
                isolate_asserts: options.z_isolate_asserts,
            };
            let cs =  ZSharpFE::gen(inputs);
            println!("    took {} ms", timer.elapsed().as_millis());

            println!("\n[+] Optimize circuit");
            let timer = Instant::now();
            let cs = opt(
                cs,
                vec![
                    Opt::ScalarizeVars,
                    Opt::Flatten,
                    Opt::Sha,
                    Opt::ConstantFold(Box::new([])),
                    Opt::Flatten,
                    Opt::Inline,
                    // Tuples must be eliminated before oblivious array elim
                    Opt::Tuple,
                    Opt::ConstantFold(Box::new([])),
                    Opt::Obliv,
                    // The obliv elim pass produces more tuples, that must be eliminated
                    Opt::Tuple,
                    Opt::LinearScan,
                    // The linear scan pass produces more tuples, that must be eliminated
                    Opt::Tuple,
                    Opt::Flatten,
                    Opt::ConstantFold(Box::new([])),
                    Opt::Inline,
                ]
            );
            println!("    took {} ms", timer.elapsed().as_millis());

            println!("\n[+] Converting to R1CS");
            let timer = Instant::now();
            let (r1cs, mut prover_data, _verifier_data) =
                to_r1cs(cs.get("main").clone(), FieldT::from(DFL_T.modulus()));
            println!("    - pre-opt R1CS size: {}", r1cs.constraints().len());
            let r1cs = reduce_linearities(r1cs, Some(options.lc_elimination_thresh));
            println!("    - final R1CS size: {}", r1cs.constraints().len());
            // save the optimized r1cs: the prover needs it to synthesize.
            prover_data.r1cs = r1cs;
            println!("    took {} ms", timer.elapsed().as_millis());

            println!("\n[+] Get instance size");
            let timer = Instant::now();
            let inputs = parse_value_map(&std::fs::read(PathBuf::from(options.pin.as_path())).unwrap());
            let (inst, _wit, _inps, num_cons, num_vars, num_inputs, num_non_zero_entries) =
                spartan::r1cs_to_spartan(&prover_data, &inputs);
            assert_ne!(num_cons, 0, "No constraints");
            println!("    took {} ms", timer.elapsed().as_millis());

            println!("\n[+] Generate public parameters");
            let timer = Instant::now();
            let gens = SNARKGens::new(num_cons, num_vars, num_inputs, num_non_zero_entries);
            println!("    took {} ms", timer.elapsed().as_millis());
        
            println!("\n[+] Encode");
            let timer = Instant::now();
            let (comm, decomm) = SNARK::encode(&inst, &gens);
            println!("    took {} ms", timer.elapsed().as_millis());

            println!("\n[+] Setup total {} ms", setup_timer.elapsed().as_millis());

            /**************************
            ******    PROVING    ******
            ***************************/

            println!("\n[+] Prove");
            let timer = Instant::now();

            let prover_input_map = parse_value_map(&std::fs::read(options.pin).unwrap());
            let (inst, wit, inps, _num_cons, _num_vars, _num_inputs, _num_non_zero_entries) =
                spartan::r1cs_to_spartan(&prover_data, &prover_input_map);
        
            let mut prover_transcript = Transcript::new(b"snark_example");
            let proof = SNARK::prove(&inst, &comm, &decomm, wit, &inps, &gens, &mut prover_transcript);
            println!("    took {} ms", timer.elapsed().as_millis());
        
            /**************************
            ******   VERIYIYNG   ******
            ***************************/

            println!("\n[+] Verify");
            let timer = Instant::now();
            // verify the proof of satisfiability
            let mut verifier_transcript = Transcript::new(b"snark_example");
            assert!(proof
              .verify(&comm, &inps, &mut verifier_transcript, &gens)
              .is_ok());        
            println!("    took {} ms", timer.elapsed().as_millis());

        }

    }

}
