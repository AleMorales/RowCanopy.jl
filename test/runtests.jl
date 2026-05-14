using Test
using RowCanopy

@testset "example_run" begin
    wheat_rows  = [RowCanopy.Hedgerow(0.15, 0.73, 3.31, 1.20) for _ in 1:2]
    wheat_inner = [RowCanopy.Hedgerow(0.15, 0.73, 2.49, 1.20) for _ in 1:2]
    maize_rows  = [RowCanopy.Hedgerow(0.40, 1.24, 1.94, 0.81) for _ in 1:2]

    rows = vcat(wheat_rows[1:1], wheat_inner, wheat_rows[2:2], maize_rows)
    intercrop = RowCanopy.StripIntercrop(rows, 135.0)

    lat = 40.9
    doy = 178
    PAR = 1200.0
    hours = collect(8.0:2:16.0)
    n_rows = length(rows)

    results = Matrix{Float64}(undef, length(hours) * n_rows, 5)
    row_out = 1
    for hour in hours
        f_dir, f_dif, f_tot = RowCanopy.instantaneous_par_interception(intercrop, PAR, lat, doy, hour)
        for i in 1:n_rows
            results[row_out, :] = [hour, Float64(i), f_dir[i], f_dif[i], f_tot[i]]
            row_out += 1
        end
    end

    @test size(results) == (length(hours)*n_rows, 5)
    @test all( (results[:,3] .>= 0.0) .& (results[:,3] .<= 1.0) )
    @test all( results[:,4] .>= 0.0 )
    @test all( results[:,5] .>= 0.0 )
end
