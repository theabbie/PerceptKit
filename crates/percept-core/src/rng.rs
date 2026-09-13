//! Deterministic RNG (xoshiro256** style) so experiments are reproducible and
//! wasm builds don't depend on `std::time`.

pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // SplitMix64 to expand a single seed into 4 lanes.
        let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = || {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        };
        Rng { s: [next(), next(), next(), next()] }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1]
            .wrapping_mul(5)
            .rotate_left(7)
            .wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform float in [0, 1).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // 24 bits of mantissa precision is plenty for init/noise.
        (self.next_u64() >> 40) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    /// Standard normal via Box-Muller.
    #[inline]
    pub fn next_normal(&mut self) -> f32 {
        let mut u1 = self.next_f32();
        if u1 < 1e-12 {
            u1 = 1e-12;
        }
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// Uniform integer in [0, n).
    #[inline]
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Shuffle a slice (Fisher-Yates).
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.next_usize(i + 1);
            xs.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_given_seed() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn normal_has_reasonable_moments() {
        let mut r = Rng::new(7);
        let n = 20_000;
        let mut mean = 0.0f32;
        let mut var = 0.0f32;
        for _ in 0..n {
            let x = r.next_normal();
            mean += x;
            var += x * x;
        }
        mean /= n as f32;
        var = var / n as f32 - mean * mean;
        assert!((mean - 0.0).abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.06, "var {var}");
    }

    #[test]
    fn shuffle_permutes() {
        let mut r = Rng::new(3);
        let mut xs: Vec<usize> = (0..64).collect();
        r.shuffle(&mut xs);
        xs.sort_unstable();
        assert_eq!(xs, (0..64).collect::<Vec<_>>());
    }
}
