use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::{Add, Sub, Mul, Div};
use wasm_bindgen::prelude::*;
use web_sys::window;
use console_error_panic_hook;

// ===========================
// DuoInt: arbitrary-precision base-12 integer
// ===========================
#[derive(Clone, Debug, PartialEq, Eq)]
struct DuoInt {
    digits: Vec<u8>, // little-endian, base-12 digits 0..11
    negative: bool,
}

impl DuoInt {
    fn normalize(mut digits: Vec<u8>, negative: bool) -> Self {
        while digits.len() > 1 && *digits.last().unwrap() == 0 {
            digits.pop();
        }
        let negative = if digits == vec![0] { false } else { negative };
        Self { digits, negative }
    }

    pub fn from_i128(mut v: i128) -> Self {
        let negative = v < 0;
        if negative {
            v = -v;
        }
        let mut digits = Vec::new();
        if v == 0 {
            digits.push(0);
        } else {
            while v > 0 {
                digits.push((v % 12) as u8);
                v /= 12;
            }
        }
        Self::normalize(digits, negative)
    }

    pub fn from_u64(mut v: u64) -> Self {
        let mut digits = Vec::new();
        if v == 0 {
            digits.push(0);
        } else {
            while v > 0 {
                digits.push((v % 12) as u8);
                v /= 12;
            }
        }
        Self::normalize(digits, false)
    }

    pub fn from_str_radix(s: &str, radix: u32) -> Result<Self, String> {
        if radix != 12 {
            return Err("Only base 12 is supported".to_string());
        }
        let mut s = s.trim().to_uppercase();
        let mut negative = false;
        if s.starts_with("-") {
            negative = true;
            s = s[1..].to_string();
        }
        if s.contains('.') {
            return Err("No fractional part for DuoInt".to_string());
        }
        if s.is_empty() {
            return Ok(Self::zero());
        }
        let mut digits: Vec<u8> = Vec::with_capacity(s.len());
        for c in s.chars().rev() {
            let val = match c {
                '0'..='9' => c as u8 - b'0',
                'A' => 10,
                'B' => 11,
                _ => return Err(format!("Invalid digit: {}", c)),
            };
            digits.push(val);
        }
        Ok(Self::normalize(digits, negative))
    }

    pub fn to_str(&self) -> String {
        self.to_str_radix(12)
    }

    pub fn to_str_radix(&self, radix: u32) -> String {
        if radix != 12 {
            panic!("Only base 12 is supported");
        }
        if self.digits == vec![0] {
            return "0".to_string();
        }
        let mut result = String::new();
        for &d in self.digits.iter().rev() {
            let ch = match d {
                0..=9 => (b'0' + d) as char,
                10 => 'A',
                11 => 'B',
                _ => unreachable!(),
            };
            result.push(ch);
        }
        if self.negative {
            result = "-".to_string() + &result;
        }
        result
    }

    fn zero() -> Self {
        Self { digits: vec![0], negative: false }
    }

    fn one() -> Self {
        Self { digits: vec![1], negative: false }
    }

    fn abs(&self) -> Self {
        let mut res = self.clone();
        res.negative = false;
        res
    }

    fn is_zero(&self) -> bool { self.digits == vec![0] }

    fn negate(&self) -> Self {
        let mut res = self.clone();
        if res.digits != vec![0] {
            res.negative = !res.negative;
        }
        res
    }

    fn multiply_by_power(&self, power: usize) -> Self {
        let mut res = self.clone();
        res.digits.splice(0..0, vec![0; power]);
        res
    }

    /// Harmonic alignment for integers (no-op except digit normalization).
    fn harmonic_align(&mut self) {
        *self = DuoInt::normalize(self.digits.clone(), self.negative);
    }
}

fn add_abs_digits(a: &Vec<u8>, b: &Vec<u8>) -> Vec<u8> {
    let max_len = a.len().max(b.len());
    let mut result: Vec<u8> = Vec::with_capacity(max_len + 1);
    let mut carry: u32 = 0;
    for i in 0..max_len {
        let sum = carry
            + if i < a.len() { a[i] as u32 } else { 0 }
            + if i < b.len() { b[i] as u32 } else { 0 };
        result.push((sum % 12) as u8);
        carry = sum / 12;
    }
    if carry > 0 {
        result.push(carry as u8);
    }
    result
}

fn sub_abs_digits(larger: &Vec<u8>, smaller: &Vec<u8>) -> Vec<u8> {
    let mut result: Vec<u8> = Vec::new();
    let mut borrow: i32 = 0;
    for i in 0..larger.len() {
        let da = larger[i] as i32 - borrow;
        let db = if i < smaller.len() { smaller[i] as i32 } else { 0 };
        let mut diff = da - db;
        if diff < 0 {
            diff += 12;
            borrow = 1;
        } else {
            borrow = 0;
        }
        result.push(diff as u8);
    }
    // Strip leading zeros
    let mut res = result;
    while res.len() > 1 && *res.last().unwrap() == 0 {
        res.pop();
    }
    if res.is_empty() {
        res.push(0);
    }
    res
}

fn cmp_abs_digits(a: &Vec<u8>, b: &Vec<u8>) -> Ordering {
    let a_len = a.len();
    let b_len = b.len();
    if a_len > b_len {
        Ordering::Greater
    } else if a_len < b_len {
        Ordering::Less
    } else {
        for i in (0..a_len).rev() {
            if a[i] > b[i] {
                return Ordering::Greater;
            } else if a[i] < b[i] {
                return Ordering::Less;
            }
        }
        Ordering::Equal
    }
}

impl Ord for DuoInt {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.negative != other.negative {
            if self.negative { Ordering::Less } else { Ordering::Greater }
        } else {
            let ord = cmp_abs_digits(&self.digits, &other.digits);
            if self.negative { ord.reverse() } else { ord }
        }
    }
}

impl PartialOrd for DuoInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Add for DuoInt {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        if self.negative == rhs.negative {
            let sum_digits = add_abs_digits(&self.digits, &rhs.digits);
            Self {
                digits: sum_digits,
                negative: self.negative,
            }
        } else {
            let cmp = cmp_abs_digits(&self.digits, &rhs.digits);
            let (larger, smaller, result_negative) = if cmp == Ordering::Greater || cmp == Ordering::Equal {
                (&self.digits, &rhs.digits, self.negative)
            } else {
                (&rhs.digits, &self.digits, rhs.negative)
            };
            let diff_digits = sub_abs_digits(larger, smaller);
            let mut result = Self {
                digits: diff_digits,
                negative: result_negative,
            };
            if result.digits == vec![0] {
                result.negative = false;
            }
            result
        }
    }
}

impl Sub for DuoInt {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + rhs.negate()
    }
}

fn mul_abs_digits(a: &Vec<u8>, b: &Vec<u8>) -> Vec<u8> {
    let mut len = a.len() + b.len();
    let mut result: Vec<u32> = vec![0; len];
    for (i, &da) in a.iter().enumerate() {
        let mut carry = 0u32;
        for (j, &db) in b.iter().enumerate() {
            let prod = result[i + j] + da as u32 * db as u32 + carry;
            result[i + j] = prod % 12;
            carry = prod / 12;
        }
        let mut k = i + b.len();
        while carry > 0 {
            if k == len {
                result.push(0);
                len += 1;
            }
            let sum = result[k] + carry;
            result[k] = sum % 12;
            carry = sum / 12;
            k += 1;
        }
    }
    let mut res: Vec<u8> = result.into_iter().map(|x| x as u8).collect();
    while res.len() > 1 && *res.last().unwrap() == 0 {
        res.pop();
    }
    if res.is_empty() {
        res.push(0);
    }
    res
}

impl Mul for DuoInt {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let digits = mul_abs_digits(&self.digits, &rhs.digits);
        let negative = self.negative != rhs.negative && digits != vec![0];
        Self { digits, negative }
    }
}

fn div_abs_digits(dividend: &Vec<u8>, divisor: &Vec<u8>) -> Vec<u8> {
    if divisor == &vec![0] {
        panic!("Division by zero");
    }
    // Long division in base-12 (little-endian digits)
    let mut quotient: Vec<u8> = vec![0; dividend.len()];
    let mut rem = DuoInt { digits: vec![0], negative: false };
    for i in (0..dividend.len()).rev() {
        // Multiply remainder by base and bring down next most-significant digit
        rem.digits.insert(0, dividend[i]);
        // Choose the largest qd in 0..=11 with (divisor * qd) <= rem
        let mut qd = 0u8;
        for d in (0..=11u8).rev() {
            let prod = DuoInt { digits: mul_abs_digits(divisor, &vec![d]), negative: false };
            if cmp_abs_digits(&prod.digits, &rem.digits) != Ordering::Greater {
                qd = d;
                break;
            }
        }
        let prod = DuoInt { digits: mul_abs_digits(divisor, &vec![qd]), negative: false };
        rem = rem - prod;
        quotient[i] = qd;
    }
    // Normalize
    let mut res = quotient;
    while res.len() > 1 && *res.last().unwrap() == 0 { res.pop(); }
    if res.is_empty() { res.push(0); }
    res
}

impl Div for DuoInt {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        if rhs == DuoInt::zero() {
            panic!("Division by zero");
        }
        let abs_q = div_abs_digits(&self.abs().digits, &rhs.abs().digits);
        let negative = self.negative != rhs.negative && abs_q != vec![0];
        Self { digits: abs_q, negative }
    }
}

// ===========================
// DuoFixed: exact dozenal fixed-point
// value / 12^scale with base-12 digits
// ===========================
#[derive(Clone, Debug, PartialEq, Eq)]
struct DuoFixed {
    value: DuoInt, // scaled integer in base-12; numeric value = value / 12^scale
    scale: usize,  // number of fractional digits (dozenal)
}

impl DuoFixed {
    pub fn new(scale: usize) -> Self {
        Self {
            value: DuoInt::zero(),
            scale,
        }
    }

    /// Construct from f64 by rounding to the nearest 12^-scale.
    pub fn from_f64(f: f64, scale: usize) -> Self {
        let neg = f.is_sign_negative();
        let scaled = (f.abs() * 12f64.powi(scale as i32)).round();
        let mut n = scaled as i128;
        let mut digits = Vec::new();
        if n == 0 {
            digits.push(0);
        } else {
            while n > 0 {
                digits.push((n % 12) as u8);
                n /= 12;
            }
        }
        Self { value: DuoInt::normalize(digits, neg), scale }
    }

    /// Parse dozenal string like "-1A.4B" into a DuoFixed with exact scale.
    pub fn from_str12(s: &str) -> Result<Self, String> {
        let s = s.trim().to_uppercase();
        if s.is_empty() { return Ok(Self::new(0)); }
        let neg = s.starts_with('-');
        let body = if neg { &s[1..] } else { &s[..] };
        let parts: Vec<&str> = body.split('.').collect();
        if parts.len() > 2 { return Err("Invalid dozenal fixed string".into()); }
        let int_part = if parts[0].is_empty() { "0" } else { parts[0] };
        let mut int_val = DuoInt::from_str_radix(int_part, 12)?;
        int_val.negative = false;

        let mut scale = 0usize;
        let mut full_digits = int_val.digits.clone();

        if parts.len() == 2 {
            let frac = parts[1];
            scale = frac.len();
            // Multiply integer by 12^scale
            full_digits.splice(0..0, vec![0; scale]);
            // Fill fractional digits into least significant positions
            let mut tmp = full_digits.clone();
            for (i, ch) in frac.chars().enumerate() {
                let d = match ch {
                    '0'..='9' => ch as u8 - b'0',
                    'A' => 10,
                    'B' => 11,
                    _ => return Err(format!("Invalid dozenal digit '{}'", ch)),
                };
                let pos = scale - 1 - i;
                tmp[pos] = d;
            }
            full_digits = tmp;
        }

        Ok(Self { value: DuoInt::normalize(full_digits, neg), scale })
    }

    /// Format to dozenal string like "-1A.4B" with the current scale.
    pub fn to_str12(&self) -> String {
        let mut digits = self.value.digits.clone();
        if digits.len() <= self.scale {
            digits.resize(self.scale + 1, 0);
        }
        let sign = if self.value.negative { "-" } else { "" };

        // integer part digits are positions >= scale
        let mut int_s = String::new();
        for &d in digits[self.scale..].iter().rev() {
            let ch = match d { 0..=9 => (b'0' + d) as char, 10 => 'A', 11 => 'B', _ => '?' };
            int_s.push(ch);
        }
        if int_s.is_empty() { int_s.push('0'); }

        if self.scale == 0 {
            format!("{sign}{int_s}")
        } else {
            let mut frac_s = String::new();
            for &d in digits[..self.scale].iter().rev() {
                let ch = match d { 0..=9 => (b'0' + d) as char, 10 => 'A', 11 => 'B', _ => '?' };
                frac_s.push(ch);
            }
            format!("{sign}{int_s}.{frac_s}")
        }
    }

    pub fn to_f64(&self) -> f64 {
        let mut val = 0f64;
        let mut power = 1f64;
        for &d in &self.value.digits {
            val += d as f64 * power;
            power *= 12.0;
        }
        let res = val / 12f64.powi(self.scale as i32);
        if self.value.negative { -res } else { res }
    }

    pub fn add(&self, other: &DuoFixed) -> DuoFixed {
        let mut res = self.clone();
        let mut oth = other.clone();
        let max_scale = self.scale.max(oth.scale);
        res.value = res.value.multiply_by_power(max_scale - self.scale);
        oth.value = oth.value.multiply_by_power(max_scale - oth.scale);
        res.value = res.value + oth.value;
        res.scale = max_scale;
        res.value.harmonic_align();
        res
    }

    pub fn mul(&self, other: &DuoFixed) -> DuoFixed {
        let mut res = self.clone();
        res.value = res.value * other.value.clone();
        res.scale += other.scale;
        res.value.harmonic_align();
        res
    }

    /// Harmonic reset: on multiples of 12, zero the least-significant fractional digit
    /// to damp micro-drift without changing the chosen precision.
    pub fn harmonic_reset(&mut self, cycle: usize) -> Self {
        if self.scale > 0 && cycle % 12 == 0 {
            if self.value.digits.len() < self.scale {
                self.value.digits.resize(self.scale, 0);
            }
            self.value.digits[0] = 0;
            self.value.harmonic_align();
        }
        self.clone()
    }
}

// ===========================
// DecInt: arbitrary-precision base-10 integer (reference VM)
// ===========================
#[derive(Clone, Debug, PartialEq, Eq)]
struct DecInt {
    digits: Vec<u8>, // little-endian, 0-9
    negative: bool,
}

impl DecInt {
    pub fn from_str_radix(s: &str, radix: u32) -> Result<Self, String> {
        if radix != 10 {
            return Err("Only base 10 is supported".to_string());
        }
        let mut s = s.trim().to_uppercase();
        let mut negative = false;
        if s.starts_with("-") {
            negative = true;
            s = s[1..].to_string();
        }
        if s.contains('.') {
            return Err("No fractional part for DecInt".to_string());
        }
        if s.is_empty() {
            return Ok(Self::zero());
        }
        let mut digits: Vec<u8> = Vec::with_capacity(s.len());
        for c in s.chars().rev() {
            let val = match c {
                '0'..='9' => c as u8 - b'0',
                _ => return Err(format!("Invalid digit: {}", c)),
            };
            digits.push(val);
        }
        // Normalize
        while digits.len() > 1 && *digits.last().unwrap() == 0 {
            digits.pop();
        }
        if digits.is_empty() {
            digits.push(0);
        }
        if digits == vec![0] {
            negative = false;
        }
        Ok(Self { digits, negative })
    }

    pub fn to_str(&self) -> String {
        self.to_str_radix(10)
    }

    pub fn to_str_radix(&self, radix: u32) -> String {
        if radix != 10 {
            panic!("Only base 10 is supported");
        }
        if self.digits == vec![0] {
            return "0".to_string();
        }
        let mut result = String::new();
        for &d in self.digits.iter().rev() {
            let ch = match d {
                0..=9 => (b'0' + d) as char,
                _ => unreachable!(),
            };
            result.push(ch);
        }
        if self.negative {
            result = "-".to_string() + &result;
        }
        result
    }

    fn zero() -> Self {
        Self { digits: vec![0], negative: false }
    }

    fn one() -> Self {
        Self { digits: vec![1], negative: false }
    }

    fn abs(&self) -> Self {
        let mut res = self.clone();
        res.negative = false;
        res
    }

    fn negate(&self) -> Self {
        let mut res = self.clone();
        if res.digits != vec![0] {
            res.negative = !res.negative;
        }
        res
    }
}

fn add_abs_digits_dec(a: &Vec<u8>, b: &Vec<u8>) -> Vec<u8> {
    let max_len = a.len().max(b.len());
    let mut result: Vec<u8> = Vec::with_capacity(max_len + 1);
    let mut carry: u32 = 0;
    for i in 0..max_len {
        let sum = carry
            + if i < a.len() { a[i] as u32 } else { 0 }
            + if i < b.len() { b[i] as u32 } else { 0 };
        result.push((sum % 10) as u8);
        carry = sum / 10;
    }
    if carry > 0 {
        result.push(carry as u8);
    }
    result
}

fn sub_abs_digits_dec(larger: &Vec<u8>, smaller: &Vec<u8>) -> Vec<u8> {
    let mut result: Vec<u8> = Vec::new();
    let mut borrow: i32 = 0;
    for i in 0..larger.len() {
        let da = larger[i] as i32 - borrow;
        let db = if i < smaller.len() { smaller[i] as i32 } else { 0 };
        let mut diff = da - db;
        if diff < 0 {
            diff += 10;
            borrow = 1;
        } else {
            borrow = 0;
        }
        result.push(diff as u8);
    }
    let mut res = result;
    while res.len() > 1 && *res.last().unwrap() == 0 {
        res.pop();
    }
    if res.is_empty() {
        res.push(0);
    }
    res
}

fn cmp_abs_digits_dec(a: &Vec<u8>, b: &Vec<u8>) -> Ordering {
    let a_len = a.len();
    let b_len = b.len();
    if a_len > b_len {
        Ordering::Greater
    } else if a_len < b_len {
        Ordering::Less
    } else {
        for i in (0..a_len).rev() {
            if a[i] > b[i] {
                return Ordering::Greater;
            } else if a[i] < b[i] {
                return Ordering::Less;
            }
        }
        Ordering::Equal
    }
}

impl Ord for DecInt {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.negative != other.negative {
            if self.negative { Ordering::Less } else { Ordering::Greater }
        } else {
            let ord = cmp_abs_digits_dec(&self.digits, &other.digits);
            if self.negative { ord.reverse() } else { ord }
        }
    }
}

impl PartialOrd for DecInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Add for DecInt {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        if self.negative == rhs.negative {
            let sum_digits = add_abs_digits_dec(&self.digits, &rhs.digits);
            Self {
                digits: sum_digits,
                negative: self.negative,
            }
        } else {
            let cmp = cmp_abs_digits_dec(&self.digits, &rhs.digits);
            let (larger, smaller, result_negative) = if cmp == Ordering::Greater || cmp == Ordering::Equal {
                (&self.digits, &rhs.digits, self.negative)
            } else {
                (&rhs.digits, &self.digits, rhs.negative)
            };
            let diff_digits = sub_abs_digits_dec(larger, smaller);
            let mut result = Self {
                digits: diff_digits,
                negative: result_negative,
            };
            if result.digits == vec![0] {
                result.negative = false;
            }
            result
        }
    }
}

impl Sub for DecInt {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + rhs.negate()
    }
}

fn mul_abs_digits_dec(a: &Vec<u8>, b: &Vec<u8>) -> Vec<u8> {
    let mut len = a.len() + b.len();
    let mut result: Vec<u32> = vec![0; len];
    for (i, &da) in a.iter().enumerate() {
        let mut carry = 0u32;
        for (j, &db) in b.iter().enumerate() {
            let prod = result[i + j] + da as u32 * db as u32 + carry;
            result[i + j] = prod % 10;
            carry = prod / 10;
        }
        let mut k = i + b.len();
        while carry > 0 {
            if k == len {
                result.push(0);
                len += 1;
            }
            let sum = result[k] + carry;
            result[k] = sum % 10;
            carry = sum / 10;
            k += 1;
        }
    }
    let mut res: Vec<u8> = result.into_iter().map(|x| x as u8).collect();
    while res.len() > 1 && *res.last().unwrap() == 0 {
        res.pop();
    }
    if res.is_empty() {
        res.push(0);
    }
    res
}

impl Mul for DecInt {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let digits = mul_abs_digits_dec(&self.digits, &rhs.digits);
        let negative = self.negative != rhs.negative && digits != vec![0];
        Self { digits, negative }
    }
}

fn div_abs_digits_dec(dividend: &Vec<u8>, divisor: &Vec<u8>) -> Vec<u8> {
    if divisor == &vec![0] {
        panic!("Division by zero");
    }
    let mut q = DecInt::zero();
    let one = DecInt::one();
    let mut rem = DecInt { digits: dividend.clone(), negative: false };
    let div = DecInt { digits: divisor.clone(), negative: false };
    while rem.cmp(&div) != Ordering::Less {
        rem = rem - div.clone();
        q = q + one.clone();
    }
    q.digits
}

impl Div for DecInt {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        if rhs == DecInt::zero() {
            panic!("Division by zero");
        }
        let abs_q = div_abs_digits_dec(&self.abs().digits, &rhs.abs().digits);
        let negative = self.negative != rhs.negative && abs_q != vec![0];
        Self { digits: abs_q, negative }
    }
}

// ===========================
// Base-12 VM (with middleware / harmonic reset)
// ===========================
#[derive(Clone)]
enum Instruction {
    MovRegImm { dst: usize, imm: DuoInt },
    MovRegReg { dst: usize, src: usize },
    Add { dst: usize, src: usize },
    Sub { dst: usize, src: usize },
    Mul { dst: usize, src: usize },
    Div { dst: usize, src: usize },
    Inc { dst: usize },
    Dec { dst: usize },
    Cmp { lhs: usize, rhs: usize },
    Je { target: usize },
    Jmp { target: usize },
    Hrst, // explicit harmonic reset
    Halt,
}

struct VM {
    registers: Vec<DuoInt>,
    pc: usize,
    program: Vec<Instruction>,
    zero_flag: bool,
    running: bool,
    instr_count: usize,
}

impl VM {
    fn new(program: Vec<Instruction>) -> Self {
        VM {
            registers: vec![DuoInt::zero(); 8],
            pc: 0,
            program,
            zero_flag: false,
            running: true,
            instr_count: 0,
        }
    }

    fn middleware_tick(&mut self) {
        self.instr_count += 1;
        if self.instr_count % 12 == 0 {
            // Normalize all registers (harmonic alignment pass)
            for r in self.registers.iter_mut() {
                r.harmonic_align();
            }
        }
    }

    pub fn run(&mut self) -> String {
        while self.running && self.pc < self.program.len() {
            let instr = self.program[self.pc].clone();
            self.pc += 1;
            match instr {
                Instruction::MovRegImm { dst, imm } => self.registers[dst] = imm,
                Instruction::MovRegReg { dst, src } => self.registers[dst] = self.registers[src].clone(),
                Instruction::Add { dst, src } => self.registers[dst] = self.registers[dst].clone() + self.registers[src].clone(),
                Instruction::Sub { dst, src } => self.registers[dst] = self.registers[dst].clone() - self.registers[src].clone(),
                Instruction::Mul { dst, src } => self.registers[dst] = self.registers[dst].clone() * self.registers[src].clone(),
                Instruction::Div { dst, src } => self.registers[dst] = self.registers[dst].clone() / self.registers[src].clone(),
                Instruction::Inc { dst } => self.registers[dst] = self.registers[dst].clone() + DuoInt::one(),
                Instruction::Dec { dst } => self.registers[dst] = self.registers[dst].clone() - DuoInt::one(),
                Instruction::Cmp { lhs, rhs } => self.zero_flag = self.registers[lhs] == self.registers[rhs],
                Instruction::Je { target } => if self.zero_flag { self.pc = target },
                Instruction::Jmp { target } => self.pc = target,
                Instruction::Hrst => {
                    for r in self.registers.iter_mut() { r.harmonic_align(); }
                }
                Instruction::Halt => self.running = false,
            }
            self.middleware_tick();
        }
        self.registers[0].to_str_radix(12)
    }
}

// ===========================
// Base-10 VM (reference, unchanged semantics)
// ===========================
#[derive(Clone)]
enum DecInstruction {
    MovRegImm { dst: usize, imm: DecInt },
    MovRegReg { dst: usize, src: usize },
    Add { dst: usize, src: usize },
    Sub { dst: usize, src: usize },
    Mul { dst: usize, src: usize },
    Div { dst: usize, src: usize },
    Inc { dst: usize },
    Dec { dst: usize },
    Cmp { lhs: usize, rhs: usize },
    Je { target: usize },
    Jmp { target: usize },
    Halt,
}

struct DecVM {
    registers: Vec<DecInt>,
    pc: usize,
    program: Vec<DecInstruction>,
    zero_flag: bool,
    running: bool,
}

impl DecVM {
    fn new(program: Vec<DecInstruction>) -> Self {
        DecVM {
            registers: vec![DecInt::zero(); 8],
            pc: 0,
            program,
            zero_flag: false,
            running: true,
        }
    }
    pub fn run(&mut self) -> String {
        while self.running && self.pc < self.program.len() {
            let instr = self.program[self.pc].clone();
            self.pc += 1;
            match instr {
                DecInstruction::MovRegImm { dst, imm } => self.registers[dst] = imm,
                DecInstruction::MovRegReg { dst, src } => self.registers[dst] = self.registers[src].clone(),
                DecInstruction::Add { dst, src } => self.registers[dst] = self.registers[dst].clone() + self.registers[src].clone(),
                DecInstruction::Sub { dst, src } => self.registers[dst] = self.registers[dst].clone() - self.registers[src].clone(),
                DecInstruction::Mul { dst, src } => self.registers[dst] = self.registers[dst].clone() * self.registers[src].clone(),
                DecInstruction::Div { dst, src } => self.registers[dst] = self.registers[dst].clone() / self.registers[src].clone(),
                DecInstruction::Inc { dst } => self.registers[dst] = self.registers[dst].clone() + DecInt::one(),
                DecInstruction::Dec { dst } => self.registers[dst] = self.registers[dst].clone() - DecInt::one(),
                DecInstruction::Cmp { lhs, rhs } => self.zero_flag = self.registers[lhs] == self.registers[rhs],
                DecInstruction::Je { target } => if self.zero_flag { self.pc = target },
                DecInstruction::Jmp { target } => self.pc = target,
                DecInstruction::Halt => self.running = false,
            }
        }
        self.registers[0].to_str_radix(10)
    }
}

// ===========================
// Assembler: base-12 VM
// ===========================
#[derive(Clone)]
enum TempInstruction {
    MovRegImm { dst: usize, imm: DuoInt },
    MovRegReg { dst: usize, src: usize },
    Add { dst: usize, src: usize },
    Sub { dst: usize, src: usize },
    Mul { dst: usize, src: usize },
    Div { dst: usize, src: usize },
    Inc { dst: usize },
    Dec { dst: usize },
    Cmp { lhs: usize, rhs: usize },
    Je { target_label: String },
    Jmp { target_label: String },
    Hrst,
    Halt,
}

fn parse_assembly(assembly: &str) -> Vec<Instruction> {
    let mut temp_program: Vec<TempInstruction> = Vec::new();
    let mut labels: HashMap<String, usize> = HashMap::new();
    let mut line_num = 0;
    for line in assembly.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(";") {
            continue;
        }
        if line.ends_with(":") {
            let label = line[0..line.len() - 1].trim().to_string();
            labels.insert(label, line_num);
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().map(|s| s.trim_end_matches(',')).collect();
        if parts.is_empty() {
            continue;
        }
        let op = parts[0].to_uppercase();
        let temp_instr = match op.as_str() {
            "MOV" => {
                if parts.len() != 3 {
                    panic!("Invalid MOV ");
                }
                let dst = parse_reg(parts[1]);
                if parts[2].starts_with("r") || parts[2].starts_with("R") {
                    let src = parse_reg(parts[2]);
                    TempInstruction::MovRegReg { dst, src }
                } else {
                    let imm = DuoInt::from_str_radix(parts[2], 12).unwrap();
                    TempInstruction::MovRegImm { dst, imm }
                }
            }
            "ADD" => {
                if parts.len() != 3 {
                    panic!("Invalid ADD ");
                }
                let dst = parse_reg(parts[1]);
                let src = parse_reg(parts[2]);
                TempInstruction::Add { dst, src }
            }
            "SUB" => {
                if parts.len() != 3 {
                    panic!("Invalid SUB ");
                }
                let dst = parse_reg(parts[1]);
                let src = parse_reg(parts[2]);
                TempInstruction::Sub { dst, src }
            }
            "MUL" => {
                if parts.len() != 3 {
                    panic!("Invalid MUL ");
                }
                let dst = parse_reg(parts[1]);
                let src = parse_reg(parts[2]);
                TempInstruction::Mul { dst, src }
            }
            "DIV" => {
                if parts.len() != 3 {
                    panic!("Invalid DIV ");
                }
                let dst = parse_reg(parts[1]);
                let src = parse_reg(parts[2]);
                TempInstruction::Div { dst, src }
            }
            "INC" => {
                if parts.len() != 2 {
                    panic!("Invalid INC ");
                }
                let dst = parse_reg(parts[1]);
                TempInstruction::Inc { dst }
            }
            "DEC" => {
                if parts.len() != 2 {
                    panic!("Invalid DEC ");
                }
                let dst = parse_reg(parts[1]);
                TempInstruction::Dec { dst }
            }
            "CMP" => {
                if parts.len() != 3 {
                    panic!("Invalid CMP ");
                }
                let lhs = parse_reg(parts[1]);
                let rhs = parse_reg(parts[2]);
                TempInstruction::Cmp { lhs, rhs }
            }
            "JE" => {
                if parts.len() != 2 {
                    panic!("Invalid JE ");
                }
                let target_label = parts[1].to_string();
                TempInstruction::Je { target_label }
            }
            "JMP" => {
                if parts.len() != 2 {
                    panic!("Invalid JMP ");
                }
                let target_label = parts[1].to_string();
                TempInstruction::Jmp { target_label }
            }
            "HRST" => {
                if parts.len() != 1 {
                    panic!("Invalid HRST ");
                }
                TempInstruction::Hrst
            }
            "HALT" => TempInstruction::Halt,
            _ => panic!("Unknown instruction: {} ", op),
        };
        temp_program.push(temp_instr);
        line_num += 1;
    }
    // Resolve labels
    let mut program: Vec<Instruction> = Vec::new();
    for temp in temp_program {
        let instr = match temp {
            TempInstruction::MovRegImm { dst, imm } => Instruction::MovRegImm { dst, imm },
            TempInstruction::MovRegReg { dst, src } => Instruction::MovRegReg { dst, src },
            TempInstruction::Add { dst, src } => Instruction::Add { dst, src },
            TempInstruction::Sub { dst, src } => Instruction::Sub { dst, src },
            TempInstruction::Mul { dst, src } => Instruction::Mul { dst, src },
            TempInstruction::Div { dst, src } => Instruction::Div { dst, src },
            TempInstruction::Inc { dst } => Instruction::Inc { dst },
            TempInstruction::Dec { dst } => Instruction::Dec { dst },
            TempInstruction::Cmp { lhs, rhs } => Instruction::Cmp { lhs, rhs },
            TempInstruction::Je { target_label } => {
                let target = *labels.get(&target_label).unwrap_or_else(|| panic!("Unknown label {} ", target_label));
                Instruction::Je { target }
            }
            TempInstruction::Jmp { target_label } => {
                let target = *labels.get(&target_label).unwrap_or_else(|| panic!("Unknown label {} ", target_label));
                Instruction::Jmp { target }
            }
            TempInstruction::Hrst => Instruction::Hrst,
            TempInstruction::Halt => Instruction::Halt,
        };
        program.push(instr);
    }
    program
}

// ===========================
// Assembler: base-10 VM (reference)
// ===========================
#[derive(Clone)]
enum TempDecInstruction {
    MovRegImm { dst: usize, imm: DecInt },
    MovRegReg { dst: usize, src: usize },
    Add { dst: usize, src: usize },
    Sub { dst: usize, src: usize },
    Mul { dst: usize, src: usize },
    Div { dst: usize, src: usize },
    Inc { dst: usize },
    Dec { dst: usize },
    Cmp { lhs: usize, rhs: usize },
    Je { target_label: String },
    Jmp { target_label: String },
    Halt,
}

fn parse_assembly_dec(assembly: &str) -> Vec<DecInstruction> {
    let mut temp_program: Vec<TempDecInstruction> = Vec::new();
    let mut labels: HashMap<String, usize> = HashMap::new();
    let mut line_num = 0;
    for line in assembly.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(";") {
            continue;
        }
        if line.ends_with(":") {
            let label = line[0..line.len() - 1].trim().to_string();
            labels.insert(label, line_num);
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().map(|s| s.trim_end_matches(',')).collect();
        if parts.is_empty() {
            continue;
        }
        let op = parts[0].to_uppercase();
        let temp_instr = match op.as_str() {
            "MOV" => {
                if parts.len() != 3 {
                    panic!("Invalid MOV ");
                }
                let dst = parse_reg(parts[1]);
                if parts[2].starts_with("r") || parts[2].starts_with("R") {
                    let src = parse_reg(parts[2]);
                    TempDecInstruction::MovRegReg { dst, src }
                } else {
                    let imm = DecInt::from_str_radix(parts[2], 10).unwrap();
                    TempDecInstruction::MovRegImm { dst, imm }
                }
            }
            "ADD" => {
                if parts.len() != 3 {
                    panic!("Invalid ADD ");
                }
                let dst = parse_reg(parts[1]);
                let src = parse_reg(parts[2]);
                TempDecInstruction::Add { dst, src }
            }
            "SUB" => {
                if parts.len() != 3 {
                    panic!("Invalid SUB ");
                }
                let dst = parse_reg(parts[1]);
                let src = parse_reg(parts[2]);
                TempDecInstruction::Sub { dst, src }
            }
            "MUL" => {
                if parts.len() != 3 {
                    panic!("Invalid MUL ");
                }
                let dst = parse_reg(parts[1]);
                let src = parse_reg(parts[2]);
                TempDecInstruction::Mul { dst, src }
            }
            "DIV" => {
                if parts.len() != 3 {
                    panic!("Invalid DIV ");
                }
                let dst = parse_reg(parts[1]);
                let src = parse_reg(parts[2]);
                TempDecInstruction::Div { dst, src }
            }
            "INC" => {
                if parts.len() != 2 {
                    panic!("Invalid INC ");
                }
                let dst = parse_reg(parts[1]);
                TempDecInstruction::Inc { dst }
            }
            "DEC" => {
                if parts.len() != 2 {
                    panic!("Invalid DEC ");
                }
                let dst = parse_reg(parts[1]);
                TempDecInstruction::Dec { dst }
            }
            "CMP" => {
                if parts.len() != 3 {
                    panic!("Invalid CMP ");
                }
                let lhs = parse_reg(parts[1]);
                let rhs = parse_reg(parts[2]);
                TempDecInstruction::Cmp { lhs, rhs }
            }
            "JE" => {
                if parts.len() != 2 {
                    panic!("Invalid JE ");
                }
                let target_label = parts[1].to_string();
                TempDecInstruction::Je { target_label }
            }
            "JMP" => {
                if parts.len() != 2 {
                    panic!("Invalid JMP ");
                }
                let target_label = parts[1].to_string();
                TempDecInstruction::Jmp { target_label }
            }
            "HALT" => TempDecInstruction::Halt,
            _ => panic!("Unknown instruction: {} ", op),
        };
        temp_program.push(temp_instr);
        line_num += 1;
    }
    // Resolve labels
    let mut program: Vec<DecInstruction> = Vec::new();
    for temp in temp_program {
        let instr = match temp {
            TempDecInstruction::MovRegImm { dst, imm } => DecInstruction::MovRegImm { dst, imm },
            TempDecInstruction::MovRegReg { dst, src } => DecInstruction::MovRegReg { dst, src },
            TempDecInstruction::Add { dst, src } => DecInstruction::Add { dst, src },
            TempDecInstruction::Sub { dst, src } => DecInstruction::Sub { dst, src },
            TempDecInstruction::Mul { dst, src } => DecInstruction::Mul { dst, src },
            TempDecInstruction::Div { dst, src } => DecInstruction::Div { dst, src },
            TempDecInstruction::Inc { dst } => DecInstruction::Inc { dst },
            TempDecInstruction::Dec { dst } => DecInstruction::Dec { dst },
            TempDecInstruction::Cmp { lhs, rhs } => DecInstruction::Cmp { lhs, rhs },
            TempDecInstruction::Je { target_label } => {
                let target = *labels.get(&target_label).unwrap_or_else(|| panic!("Unknown label {} ", target_label));
                DecInstruction::Je { target }
            }
            TempDecInstruction::Jmp { target_label } => {
                let target = *labels.get(&target_label).unwrap_or_else(|| panic!("Unknown label {} ", target_label));
                DecInstruction::Jmp { target }
            }
            TempDecInstruction::Halt => DecInstruction::Halt,
        };
        program.push(instr);
    }
    program
}

fn parse_reg(s: &str) -> usize {
    let s = s.to_lowercase();
    if s.starts_with("r") {
        s[1..].parse::<usize>().unwrap()
    } else {
        panic!("Invalid register: {} ", s);
    }
}

// ===========================
// WASM exports
// ===========================
#[wasm_bindgen]
pub fn run_assembly(assembly: &str) -> String {
    let program = parse_assembly(assembly);
    let mut vm = VM::new(program);
    vm.run()
}

#[wasm_bindgen]
pub fn add(a: &str, b: &str) -> String {
    let x = DuoInt::from_str_radix(a, 12).unwrap();
    let y = DuoInt::from_str_radix(b, 12).unwrap();
    let sum = x + y;
    sum.to_str_radix(12)
}

/// Convert f64 → dozenal fixed string with `scale` fractional digits.
#[wasm_bindgen]
pub fn to_duodecimal(value: f64, scale: usize) -> String {
    DuoFixed::from_f64(value, scale).to_str12()
}

/// Parse dozenal string like "1A.4" → f64 (returns 0.0 on parse error).
#[wasm_bindgen]
pub fn from_duodecimal(s: &str) -> f64 {
    DuoFixed::from_str12(s).map(|d| d.to_f64()).unwrap_or(0.0)
}

#[wasm_bindgen]
pub fn tune_weights(weights: Vec<f64>, scale: usize) -> Vec<f64> {
    let mut tuned = Vec::new();
    for &w in &weights {
        let mut df = DuoFixed::from_f64(w, scale);
        df = df.harmonic_reset(12); // apply reset as if on the 12th cycle
        tuned.push(df.to_f64());
    }
    tuned
}

/// Like `tune_weights` but allows specifying the current cycle.
#[wasm_bindgen]
pub fn tune_weights_at_cycle(weights: Vec<f64>, scale: usize, cycle: usize) -> Vec<f64> {
    let mut tuned = Vec::new();
    for &w in &weights {
        let mut df = DuoFixed::from_f64(w, scale);
        df = df.harmonic_reset(cycle);
        tuned.push(df.to_f64());
    }
    tuned
}

#[wasm_bindgen]
pub fn evaluate_drift(weights: Vec<f64>, iterations: usize, scale: usize, with_tuning: bool) -> f64 {
    let performance = window().unwrap().performance().unwrap();
    let start = performance.now();
    let mut current = weights.clone();
    for i in 1..=iterations {
        // Simulate drift by adding small noise
        for w in current.iter_mut() {
            *w += 0.001 * (i as f64); // Accumulating noise
        }
        if with_tuning && i % 12 == 0 {
            current = tune_weights_at_cycle(current, scale, i);
        }
    }
    let end = performance.now();
    end - start
}

// ===========================
// Bench/demo programs
// ===========================
const ASSEMBLY_WITHOUT: &str = r#"
mov r0, 0
mov r1, 1
mov r2, 100001
loop:
cmp r1, r2
je end
add r0, r1
inc r1
jmp loop
end:
halt
"#;

const ASSEMBLY_WITH: &str = r#"
mov r0, 0
mov r1, 1
mov r2, 49A55
loop:
cmp r1, r2
hrst
je end
add r0, r1
inc r1
jmp loop
end:
halt
"#;

#[wasm_bindgen]
pub fn evaluate_without_middleware() -> f64 {
    let performance = window().unwrap().performance().unwrap();
    let start = performance.now();
    let program = parse_assembly_dec(ASSEMBLY_WITHOUT);
    let mut vm = DecVM::new(program);
    let _ = vm.run();
    let end = performance.now();
    end - start
}

#[wasm_bindgen]
pub fn evaluate_with_middleware() -> f64 {
    let performance = window().unwrap().performance().unwrap();
    let start = performance.now();
    let program = parse_assembly(ASSEMBLY_WITH);
    let mut vm = VM::new(program);
    let _ = vm.run();
    let end = performance.now();
    end - start
}

#[wasm_bindgen(start)]
pub fn main() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    Ok(())
}