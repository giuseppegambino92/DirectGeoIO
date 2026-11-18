//! WMO GRIB2 Code Table 4.2 (parameter) and 4.5 (fixed surface) lookup for the common
//! meteorological variables. Unknown entries are reported generically rather than guessed.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Param {
    pub short_name: &'static str,
    pub name: &'static str,
    pub units: &'static str,
}

const UNKNOWN: Param = Param { short_name: "unknown", name: "unknown", units: "unknown" };

/// (discipline, category, number) -> parameter. Discipline 0 = meteorological products,
/// discipline 2 = land surface, discipline 10 = oceanographic.
pub fn lookup(discipline: u8, category: u8, number: u8) -> Param {
    let p = |s, n, u| Param { short_name: s, name: n, units: u };
    match (discipline, category, number) {
        (0, 0, 0) => p("t", "Temperature", "K"),
        (0, 0, 2) => p("pt", "Potential temperature", "K"),
        (0, 0, 4) => p("tmax", "Maximum temperature", "K"),
        (0, 0, 5) => p("tmin", "Minimum temperature", "K"),
        (0, 0, 6) => p("dpt", "Dew point temperature", "K"),
        (0, 1, 0) => p("q", "Specific humidity", "kg kg-1"),
        (0, 1, 1) => p("r", "Relative humidity", "%"),
        (0, 1, 3) => p("pwat", "Precipitable water", "kg m-2"),
        (0, 1, 7) => p("prate", "Precipitation rate", "kg m-2 s-1"),
        (0, 1, 8) => p("tp", "Total precipitation", "kg m-2"),
        (0, 1, 11) => p("sde", "Snow depth", "m"),
        (0, 1, 13) => p("sdwe", "Water equivalent of accumulated snow depth", "kg m-2"),
        (0, 1, 52) => p("tprate", "Total precipitation rate", "kg m-2 s-1"),
        (0, 1, 193) => p("tp", "Total precipitation (ECMWF local)", "m"),
        (0, 2, 0) => p("wdir", "Wind direction", "degree"),
        (0, 2, 1) => p("ws", "Wind speed", "m s-1"),
        (0, 2, 2) => p("u", "U component of wind", "m s-1"),
        (0, 2, 3) => p("v", "V component of wind", "m s-1"),
        (0, 2, 8) => p("w", "Vertical velocity (pressure)", "Pa s-1"),
        (0, 2, 10) => p("absv", "Absolute vorticity", "s-1"),
        (0, 2, 22) => p("gust", "Wind speed (gust)", "m s-1"),
        (0, 3, 0) => p("pres", "Pressure", "Pa"),
        (0, 3, 1) => p("prmsl", "Pressure reduced to MSL", "Pa"),
        (0, 3, 4) => p("z", "Geopotential", "m2 s-2"),
        (0, 3, 5) => p("gh", "Geopotential height", "gpm"),
        (0, 6, 1) => p("tcc", "Total cloud cover", "%"),
        (0, 7, 6) => p("cape", "Convective available potential energy", "J kg-1"),
        (0, 19, 0) => p("vis", "Visibility", "m"),
        (2, 0, 0) => p("lsm", "Land cover (1=land, 0=sea)", "Proportion"),
        (2, 0, 2) => p("st", "Soil temperature", "K"),
        (2, 0, 192) => p("soilw", "Volumetric soil moisture content (NCEP local)", "Fraction"),
        (2, 3, 18) => p("st", "Soil temperature", "K"),
        (2, 3, 20) => p("soilw", "Soil moisture", "kg m-3"),
        (10, 2, 0) => p("ci", "Sea ice area fraction", "Proportion"),
        (10, 3, 0) => p("sst", "Sea surface temperature", "K"),
        _ => UNKNOWN,
    }
}

/// Code Table 4.5 (type of fixed surface), common entries.
pub fn level_type_name(t: u8) -> &'static str {
    match t {
        1 => "surface",
        6 => "maxWind",
        7 => "tropopause",
        100 => "isobaricInPa",
        101 => "meanSea",
        102 => "heightAboveSea",
        103 => "heightAboveGround",
        104 => "sigma",
        105 => "hybrid",
        106 => "depthBelowLand",
        150 => "generalVertical",
        151 => "soil",
        160 => "depthBelowSea",
        200 => "entireAtmosphere",
        _ => "unknown",
    }
}
