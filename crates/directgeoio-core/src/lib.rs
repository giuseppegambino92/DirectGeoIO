//! # directgeoio-core
//!
//! GRIB2 section scanning, WMO parameter tables, bit-exact CPU reference decoders
//! (Templates 5.0, 5.2, 5.3, 5.42) and the device-independent decode plan that the
//! Metal and CUDA backends execute.

pub mod aec;
pub mod backend;
pub mod bits;
pub mod cpu;
pub mod decode;
pub mod grib2;
pub mod npy;
pub mod param;
pub mod plan;

pub use grib2::{scan, ByteRange, Error, Field, Packing, Result};
