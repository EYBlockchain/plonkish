// TO DO: don't want to use two lines below
use halo2_proofs::arithmetic::Field;
use super::{Mds, Spec};
use halo2_curves::bn256::Fr;
use crate::circuits::poseidongadget::poseidon::primitives::generate_constants;

// To do rewrite this below
/// Poseidon-128 using the $x^5$ S-box, with a width of 3 field elements, and the
/// standard number of rounds for 128-bit security "with margin".
///
/// The standard specification for this set of parameters (on either of the Pasta
/// fields) uses $R_F = 8, R_P = 56$. This is conveniently an even number of
/// partial rounds, making it easier to construct a Halo 2 circuit.
#[derive(Debug)]
// Do we have to specify width and rate generically?
pub struct BN256param<const T: usize, const R: usize,const SECURE_MDS: usize>;

impl<const T: usize, const R: usize, const SECURE_MDS: usize > Spec<Fr, T, R> for BN256param<T,R,SECURE_MDS> {
    fn full_rounds() -> usize {
        8
    }

    fn partial_rounds() -> usize {
        //TO DO: we need an even number of partial rounds - can we round up
        match T {
            2 => 56,
            // Rounded up from 57
            3 => 58,
            4 => 56,
            5 => 60,
            6 => 60,
            // Rounded up from 63
            7 => 64,
            _ => unimplemented!(),
        }
    }

    fn sbox(val: Fr) -> Fr {
        val.pow_vartime([5])
    }

    fn secure_mds() -> usize {
        SECURE_MDS
    }

    fn constants() -> (Vec<[Fr; T]>, Mds<Fr, T>, Mds<Fr, T>) {
        // TO DO: manually generate the constants here
        generate_constants::<_, Self, T, R>()
    }
}

// TO DO Remove both


#[cfg(test)]
mod tests {
    #![allow(dead_code)]
    use crate::circuits::poseidongadget::poseidon::primitives::{
        generate_constants, permute, ConstantLength, Hash, Mds, Spec,
    };
    use ff::PrimeField;
    use ff::{Field, FromUniformBytes};
    use std::marker::PhantomData;

    /// The same Poseidon specification as poseidon::P128Pow5T3, but constructed
    /// such that its constants will be generated at runtime.
    #[derive(Debug)]
    // to do change Field to Fr?
    pub struct BN256paramGen<const T: usize, const R: usize, F: Field, const SECURE_MDS: usize>(PhantomData<F>);

    impl<const T: usize, const R: usize, F: Field, const SECURE_MDS: usize> BN256paramGen<T, R, F, SECURE_MDS> {
        pub fn new() -> Self {
            BN256paramGen(PhantomData)
        }
    }

    impl<const T: usize, const R: usize, F: FromUniformBytes<64> + Ord, const SECURE_MDS: usize> Spec<F, T, R> for BN256paramGen<T,R, F, SECURE_MDS> {
        fn full_rounds() -> usize {
            8
        }
    
        fn partial_rounds() -> usize {
            //TO DO: we need an even number of partial rounds - can we round up
            match T {
                2 => 56,
                // Rounded up from 57
                3 => 58,
                4 => 56,
                5 => 60,
                6 => 60,
                // Rounded up from 63
                7 => 64,
                _ => unimplemented!(),
            }
        }
    
        fn sbox(val: F) -> F {
            val.pow_vartime([5])
        }
    
        fn secure_mds() -> usize {
            SECURE_MDS
        }
    
        fn constants() -> (Vec<[F; T]>, Mds<F, T>, Mds<F, T>) {
            // TO DO: manually generate the constants here
            generate_constants::<_, Self, T, R>()
        }
    }

    #[test]
    #[ignore]
    fn verify_constants() {
        // TO DO: write this as in p128pow5t3.rs once we have the constants manually generated
    }

    #[test]
    #[ignore]
    fn test_against_reference() {
       // TO DO: write this  as in p128pow5t3.rs once we have the constants manually generated
    }

    #[test]
    #[ignore]
    fn permute_test_vectors() {

       /*  macro_rules! permute_test_vectors {
            ($t:expr) => {
                {
                    r = $t - 1;
                    let (round_constants, mds, _) = super::BN256param::<$t,r>::constants();

                    // Generate test vectors for the permutation
                    for tv in crate::frontend::halo2::poseidongadget::poseidon::primitives::test_vectors::fp::permute() {
                        let mut state = [
                            Fp::from_repr(tv.initial_state[0]).unwrap(),
                            Fp::from_repr(tv.initial_state[1]).unwrap(),
                            Fp::from_repr(tv.initial_state[2]).unwrap(),
                        ];

                        permute::<Fr, super::BN256param<$t,r>, $t, r>(&mut state, &mds, &round_constants);

                        for (expected, actual) in tv.final_state.iter().zip(state.iter()) {
                            assert_eq!(&actual.to_repr(), expected);
                        }
                    }
                }

            };
        }

        for t in 1..7 {
            permute_test_vectors!(t);
            
        }*/

        
    }

    #[test]
    #[ignore]
     // Generate test vectors for the permutation and adapt this
    fn hash_test_vectors() {
    }
}
