from lightguide import restitution as res

from pyrocko import trace


def test_strainrate_to_strain(syn_das_data, show_plot):
    syn_e = syn_das_data(quantity="strain", sample_rate=5.0, smoothing_sigma=100.0)
    syn_de = syn_das_data(
        quantity="strain_rate", sample_rate=5.0, smoothing_sigma=100.0
    )

    res_e = res.strainrate_to_strain(syn_de, copy=True)

    # num.testing.assert_equal(
    #     [tr.ydata for tr in res_e],
    #     [tr.ydata for tr in syn_e]
    # )
    for tr in res_e:
        tr.set_location("RES")
    if show_plot:
        trace.snuffle(syn_e + res_e)
