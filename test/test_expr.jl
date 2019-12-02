import AdversarialPrediction: CM_TP, CM_FP, CM_FN, CM_TN, CM_AP, CM_AN, CM_PP, CM_PN, CM_ALL

tp = CM_TP()
fp = CM_FP() 
fn = CM_FN() 
tn = CM_TN() 
ap = CM_AP()
an = CM_AN() 
pp = CM_PP()
pn = CM_PN()
al = CM_ALL()


@testset "expression" begin
    @test string(2 * tp) == "2.0 TP"
    @test string(tp + fp) == "PP"
    @test string(ap + an) == "ALL"
    @test string(ap + pp) == "(AP) + (PP)"
    @test string(2 * tp / (ap + pp)) == "(2.0 TP) / ((AP) + (PP))" 

    kappa = ((tp + tn) / al - (ap * pp + an * pn) / al^2) / (1 - (ap * pp + an * pn) / al^2)
    str_kappa = "((((TP) + (TN)) / (ALL)) + (-((((AP) * (PP)) + ((AN) * (PN))) / ((ALL)^2.0)))) / ((1.0) + (-((((AP) * (PP)) + ((AN) * (PN))) / ((ALL)^2.0))))"
    @test string(kappa) == str_kappa
end


@testset "expression info" begin
    acc = (tp + tn) / (ap + an)
    @test acc.info.needs_adv_sum_marg == false

    prec = tp / pp
    @test prec.info.needs_pred_sum_marg == true
    @test prec.info.needs_adv_sum_marg == false

    gm = tp / (sqrt(pp * ap))
    @test gm.info.needs_adv_sum_marg == true
    @test gm.info.needs_pred_sum_marg == true
    @test gm.info.is_linear_tp_tn == true

    kappa = ((tp + tn) / al - (ap * pp + an * pn) / al^2) / (1 - (ap * pp + an * pn) / al^2)
    @test kappa.info.needs_pred_sum_marg == true
    @test kappa.info.needs_adv_sum_marg == true
    @test kappa.info.is_linear_tp_tn == true

    gm_err = tp / (sqrt(pp * tn))
    @test gm_err.info.needs_pred_sum_marg == true
    @test gm_err.info.needs_adv_sum_marg == false
    @test gm_err.info.is_linear_tp_tn == false
end


@testset "invalid metric" begin
    @metric Metric1
    function define(::Type{Metric1}, C::ConfusionMatrix)
        return C.tp * C.tn / sqrt(C.ap * C.pp)
    end
    @metric Metric2
    function define(::Type{Metric2}, C::ConfusionMatrix)
        return C.ap / C.all
    end
    @metric Metric3
    function define(::Type{Metric3}, C::ConfusionMatrix)
        return (C.tp / C.all) * (1 - C.fp) / (C.pp)
    end
    @metric Metric4
    function define(::Type{Metric4}, C::ConfusionMatrix)
        return C.tp / (C.ap + C.pp + C.fp)
    end

    @test_throws ErrorException Metric1()
    @test_throws ErrorException Metric2()
    @test_throws ErrorException Metric3()
    @test_throws ErrorException Metric4()
end

