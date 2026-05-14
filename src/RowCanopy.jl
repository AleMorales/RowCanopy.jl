# Adapted from the model by Gijzen & Goudriaan (1989) and Wang et al. (2017)
# Light interception model for strip intercropping systems

module RowCanopy

using LinearAlgebra
using SolarPosition
using Dates
using Printf
# ============================================================
# DATA STRUCTURES
# ============================================================

"""
    Hedgerow represents a single crop row as a rectangular hedgerow
"""
struct Hedgerow
    width::Float64      # wH: hedgerow cross-section width (m)
    height::Float64     # h: canopy height (m)
    LAI::Float64        # leaf area index (m2 m-2)
    LAD::Float64        # leaf area density (m2 m-3), derived from LAI
    chi::Float64        # ratio of vertical to horizontal projection of canopy elements
    row_spacing::Float64  # wR: inter-row spacing (m)
end

"""
Constructor for Hedgerow that computes LAD and hedgerow width from LAI and ground cover fraction
    Sources: Wang 2.1, eq 4 (tau_c calculation described in text)
"""
function Hedgerow(row_spacing::Float64, height::Float64, LAI::Float64, chi::Float64;
                  LAI_max_cover::Float64=3.0)
    tau_c = min(LAI / LAI_max_cover, 1.0)   # fraction of ground cover
    width = row_spacing * tau_c              # wH (m)
    LAD = (width > 0 && height > 0) ? LAI * row_spacing / (width * height) : 0.0
    return Hedgerow(width, height, LAI, LAD, chi, row_spacing)
end

"""
    StripIntercrop holds the full description of an intercropping strip
    rows: vector of Hedgerow (ordered left to right)
    row_direction: azimuth of row direction from north (degrees)
"""
struct StripIntercrop
    rows::Vector{Hedgerow}
    row_direction::Float64   # degrees from north
    strip_width::Float64     # wS: total width of one complete intercropping strip (m)
end

function StripIntercrop(rows::Vector{Hedgerow}, row_direction::Float64)
    strip_width = sum(r.row_spacing for r in rows)
    return StripIntercrop(rows, row_direction, strip_width)
end

# ============================================================
# SOLAR GEOMETRY
# ============================================================

"""
    solar_position(lat, doy, hour) -> (elevation_deg, azimuth_deg)

Compute solar elevation (beta) and azimuth (from north, positive eastward) angles in degrees
for a given latitude (degrees), day of year, and solar hour.
Uses PSA algorithm from SolarPosition.jl
"""
function solar_position(lat::Float64, doy::Int, hour::Float64)
    dt  = DateTime(2025, 1, 1) + Day(doy - 1) + Millisecond(round(Int, hour * 3600 * 1000))
    obs = Observer(lat, 0.0, 0.0)
    angles = SolarPosition.solar_position(obs, dt)
    azimuth = angles.azimuth
    elevation = 90.0 - angles.zenith
    return (elevation, azimuth)
end

"""
    par_partition(PAR_total, doy, hour, lat) -> (PAR_direct, PAR_diffuse)

Partition total incident PAR into direct and diffuse components.
Based on Spitters et al. (1986) and De Jong (1980).
"""
function par_partition(PAR_total::Float64, doy::Int, hour::Float64, lat::Float64)
    elev_deg, _ = solar_position(lat, doy, hour)
    if elev_deg <= 0.0
        return 0.0, 0.0
    end
    elev = deg2rad(elev_deg)
    # Solar constant corrected for Earth-Sun distance variation using equation of time approximation by Spencer (1971)
    # (Note: the same as in Spitters et al. 1986, but with three extra terms for the eccentricity E0)
    B  = 2π * (doy - 1) / 365.0
    E0 = 1.000110 + 0.034221 * cos(B) + 0.001280 * sin(B) +
                    0.000719 * cos(2B) + 0.000077 * sin(2B)
    S0 = 1367.0 * E0 * 0.5   # solar constant, ~50% as PAR (W m-2)

    # Extra-terrestrial PAR (W m-2)
    PAR_et = S0 * sin(elev)
    PAR_et = max(PAR_et, 1e-6)
    tau = PAR_total / PAR_et   # atmospheric transmission

    # Hourly diffuse fraction (Spitters et al. 1986, Eq. 2b)
    # Note: depends on both tau and solar elevation
    R = 0.847 - 1.61 * sin(elev) + 1.04 * sin(elev)^2    # clear-sky limit
    K = (1.47 - R) / 1.66                                # transition point
    frac_diffuse = if tau <= 0.22
        1.0
    elseif tau <= 0.35
        1.0 - 6.4 * (tau - 0.22)^2
    elseif tau <= K
        1.47 - 1.66 * tau
    else
        R
    end
    frac_diffuse = clamp(frac_diffuse, 0.0, 1.0)
    PAR_diffuse = frac_diffuse * PAR_total
    PAR_direct  = PAR_total - PAR_diffuse
    return PAR_direct, PAR_diffuse
end

# ============================================================
# GEOMETRY: beam angles (Gijzen & Goudriaan 1989, Wang et al. 2017)
# ============================================================

"""
    beam_angles(solar_elev_deg, solar_az_deg, row_direction_deg)
    -> (theta_b, theta_c)

Compute the angle of the beam within the xz-plane (theta_b) and the
angle between the vertical plane through zenith+beam and the xz-plane (theta_c).
Equations (6) and (7) in Wang et al. (2017), equivalent to (1)-(2) in Gijzen & Goudriaan (1989).
See Figure 1 in Wang et al. (2017) for geometry.
"""
function beam_angles(solar_elev_deg::Float64, solar_az_deg::Float64,
                     row_direction_deg::Float64)
    beta  = deg2rad(solar_elev_deg)
    theta_a = deg2rad(row_direction_deg - solar_az_deg)

    sin_theta_c = cos(π - theta_a) * cos(beta)   # Eq. 6
    sin_theta_c = clamp(sin_theta_c, -1.0, 1.0)
    theta_c = asin(sin_theta_c)

    cos_theta_c = cos(theta_c)
    if abs(cos_theta_c) < 1e-10
        theta_b = π/2
    else
        cos_theta_b_from_eq7 = cos(theta_c) > 0 ? sin(beta) / cos(theta_c) : 0.0
        cos_theta_b_from_eq7 = clamp(cos_theta_b_from_eq7, -1.0, 1.0)
        theta_b = acos(cos_theta_b_from_eq7)           # Eq. 7
    end
    return theta_b, theta_c   # both in radians
end

# ============================================================
# GEOMETRY: transmission distance through hedgerows (Wang et al. 2017)
# ============================================================

"""
    transmission_distance(intercrop, row_idx, theta_b, d0)
    -> d_prime (horizontal transmission distance in xz-plane for hedgerow i)

Implements Eq. 10-12 from Wang et al. (2017).
d0: horizontal range between end of beam and right side of last unit strip (m)
row_idx: 1-based index of target hedgerow
"""
function transmission_distance(intercrop::StripIntercrop, row_idx::Int,
                                theta_b::Float64, d0::Float64,
                                hmax::Float64)
    rows = intercrop.rows
    n_rows = length(rows)
    wS = intercrop.strip_width

    # Integer number of complete strips traversed (Eq. 8)
    N = floor(Int, (d0 + hmax * tan(theta_b)) / wS)

    # Which row within a strip is the beam incident on (row_idx within strip)
    m = n_rows
    a = div(row_idx - 1, m)   # integer part
    b = mod(row_idx - 1, m)   # remainder (0-based within strip)

    # Cumulative row widths within strip up to row b
    cum_width(k) = sum(rows[j].row_spacing for j in 1:k; init=0.0)

    # h'_i and h''_i: heights where beam exits and enters hedgerow i (Eqs. 11-12)
    wRb = rows[b+1].row_spacing
    hb  = rows[b+1].height

    h_exit  = (a * wS + cum_width(b) + 0.5 * (wRb - hb) - d0) / tan(theta_b + 1e-10)
    h_enter = (a * wS + cum_width(b) + 0.5 * (wRb + hb) - d0) / tan(theta_b + 1e-10)

    hi = rows[row_idx > n_rows ? n_rows : row_idx].height
    wHi = rows[row_idx > n_rows ? n_rows : row_idx].width

    # d'_i from Eq. 10
    d_prime = if h_exit < 0
        0.0
    elseif h_exit <= 0 && h_enter >= 0
        h_exit * tan(theta_b)
    elseif h_enter > 0 && h_exit > 0 && hi <= h_exit
        0.0
    elseif h_enter > 0 && h_exit > 0 && hi > h_exit
        (hi - h_exit) * tan(theta_b)
    else
        wHi
    end

    return max(d_prime, 0.0)
end

# ============================================================
# CANOPY EXTINCTION COEFFICIENT (Campbell & Norman 1998, Eq. 3 in Wang et al. 2017)
# ============================================================

"""
    extinction_coeff(chi, zenith_rad) -> g

G-function based extinction coefficient.
chi: ratio of vertical to horizontal projection of canopy elements.
zenith_rad: solar zenith angle (radians).
"""
function extinction_coeff(chi::Float64, zenith_rad::Float64)
    psi = zenith_rad
    g = sqrt(chi^2 * cos(psi)^2 + sin(psi)^2) /
        (chi + 1.774 * (chi + 1.182)^(-0.733))
    return g
end

# ============================================================
# LIGHT INTERCEPTION PER HEDGEROW (Beer's Law, Eqs. 1-2 Wang et al. 2017)
# ============================================================

"""
    light_interception_direct(intercrop, PAR_direct, solar_elev_deg, solar_az_deg)
    -> Vector of fractional direct PAR interception per row

For a direct beam, computes fraction intercepted by each hedgerow.
Integration step: dx = 0.1 m across hedgerow cross-section.
"""
function light_interception_direct(intercrop::StripIntercrop,
                                    PAR_direct::Float64,
                                    solar_elev_deg::Float64,
                                    solar_az_deg::Float64)
    rows = intercrop.rows
    n = length(rows)
    f = zeros(Float64, n)

    if solar_elev_deg <= 0.0 || PAR_direct <= 0.0
        return f
    end

    theta_b, theta_c = beam_angles(solar_elev_deg, solar_az_deg,
                                    intercrop.row_direction)
    zenith = π/2 - deg2rad(solar_elev_deg)
    hmax = maximum(r.height for r in rows)
    wS   = intercrop.strip_width

    # Integrate across horizontal positions within one strip (dx = 0.1 m)
    dx = 0.1
    xs  = 0.0:dx:wS
    n_pts = length(xs)
    f_sum = zeros(Float64, n)

    for x in xs
        d0 = x   # horizontal position as proxy for d0
        cumulative_exp = 0.0   # sum of g*LAD*d for all preceding hedgerows
        for i in 1:n
            g  = extinction_coeff(rows[i].chi, zenith)
            dp = transmission_distance(intercrop, i, theta_b, d0, hmax)
            # actual path length through hedgerow (Eq. 5)
            d  = (abs(sin(theta_b)) > 1e-6 && abs(cos(theta_c)) > 1e-6) ?
                  dp / (sin(theta_b) * cos(theta_c)) : 0.0
            gLd = g * rows[i].LAD * d
            # fraction intercepted by this hedgerow (Eq. 1)
            fi = exp(-cumulative_exp) * (1.0 - exp(-gLd))
            f_sum[i] += fi
            cumulative_exp += gLd
        end
    end

    for i in 1:n
        f[i] = f_sum[i] / n_pts
    end
    return f
end

"""
    light_interception_diffuse(intercrop, PAR_diffuse)
    -> Vector of fractional diffuse PAR interception per row

Approximates diffuse radiation with 324 directional sources (9 elevation circles x 36 azimuths).
"""
function light_interception_diffuse(intercrop::StripIntercrop,
                                     PAR_diffuse::Float64)
    rows = intercrop.rows
    n = length(rows)
    f_total = zeros(Float64, n)

    if PAR_diffuse <= 0.0
        return f_total
    end

    n_elev = 9
    n_az   = 36
    total_sources = n_elev * n_az

    for ie in 1:n_elev
        elev_deg = (ie - 0.5) * (90.0 / n_elev)   # centre of elevation band
        weight   = cos(deg2rad(elev_deg))           # weight by cos(zenith) for isotropic sky
        for ia in 1:n_az
            az_deg = (ia - 0.5) * (360.0 / n_az)
            fi = light_interception_direct(intercrop, 1.0, elev_deg, az_deg)
            for i in 1:n
                f_total[i] += weight * fi[i]
            end
        end
    end

    # Normalize
    norm = sum(cos(deg2rad((ie - 0.5) * 90.0 / n_elev)) for ie in 1:n_elev) * n_az
    for i in 1:n
        f_total[i] = f_total[i] / norm * PAR_diffuse
    end
    return f_total
end

# ============================================================
# TOP-LEVEL: instantaneous PAR interception
# ============================================================

"""
    instantaneous_par_interception(intercrop, PAR_total, lat, doy, hour)
    -> (f_direct, f_diffuse, f_total) each a Vector of length n_rows

Returns fraction of above-canopy PAR intercepted by each row.
"""
function instantaneous_par_interception(intercrop::StripIntercrop,
                                         PAR_total::Float64,
                                         lat::Float64, doy::Int,
                                         hour::Float64)
    elev_deg, az_deg = solar_position(lat, doy, hour)
    PAR_direct, PAR_diffuse = par_partition(PAR_total, doy, hour, lat)

    f_dir  = light_interception_direct(intercrop, PAR_direct, elev_deg, az_deg)
    f_dif  = light_interception_diffuse(intercrop, PAR_diffuse)

    n = length(intercrop.rows)
    f_tot  = [f_dir[i] + f_dif[i] for i in 1:n]

    return f_dir, f_dif, f_tot
end

end\n
