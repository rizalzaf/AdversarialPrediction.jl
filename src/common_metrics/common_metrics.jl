import ..AdversarialPrediction: @metric, define, constraint, ConfusionMatrix,
    special_case_positive!, special_case_negative!, 
    cs_special_case_positive!, cs_special_case_negative!

# metrics
@metric Accuracy
function define(::Type{Accuracy}, C::ConfusionMatrix)
    return (C.tp + C.tn) / (C.all)  
end   

accuracy_metric = Accuracy()


@metric Precision       # Precision
function define(::Type{Precision}, C::ConfusionMatrix)
    return C.tp / C.pp
end   
prec = Precision()
special_case_positive!(prec)


@metric Recall       # Recall / Sensitivity
function define(::Type{Recall}, C::ConfusionMatrix)
    return C.tp / C.ap
end   
rec = Recall()
special_case_positive!(rec)

@metric Specificity       # Specificity
function define(::Type{Specificity}, C::ConfusionMatrix)
    return C.tn / C.an
end   
spec = Specificity()
special_case_negative!(spec)


# F1
@metric F1Score
function define(::Type{F1Score}, C::ConfusionMatrix)
    return (2 * C.tp) / (C.ap + C.pp)  
end   

f1_score = F1Score()
special_case_positive!(f1_score)


# metric with arguments
@metric FBeta beta
function define(::Type{FBeta}, C::ConfusionMatrix, beta)
    return ((1 + beta^2) * C.tp) / (beta^2 * C.ap + C.pp)  
end   

f2_score = FBeta(2.0)
special_case_positive!(f2_score)

fhalf_score = FBeta(0.5)
special_case_positive!(fhalf_score)


# Geometric Mean of Prec and Rec
@metric GM_PrecRec
function define(::Type{GM_PrecRec}, C::ConfusionMatrix)
    return C.tp / sqrt(C.ap * C.pp)  
end   

gpr = GM_PrecRec()
special_case_positive!(gpr)


# informedness
@metric Informedness 
function define(::Type{Informedness}, C::ConfusionMatrix)
    return C.tp / C.ap + C.tn / C.an - 1
end   

inform = Informedness()
special_case_positive!(inform)
special_case_negative!(inform)


@metric Kappa
function define(::Type{Kappa}, C::ConfusionMatrix)
    pe = (C.ap * C.pp + C.an * C.pn) / C.all^2
    num = (C.tp + C.tn) / C.all - pe
    den = 1 - pe
    return num / den
end  

kappa = Kappa()
special_case_positive!(kappa)
special_case_negative!(kappa)


@metric MCC
function define(::Type{MCC}, C::ConfusionMatrix)
    num = C.tp / C.all - (C.ap * C.pp) / C.all^2
    den = sqrt(C.ap * C.pp * C.an * C.pn) / C.all^2
    return num / den
end  

mcc = MCC()
special_case_positive!(mcc)
special_case_negative!(mcc)



########## METRIC WITH CONSTRAINTS

# precision given recall
@metric PrecisionGvRecall th
function define(::Type{PrecisionGvRecall}, C::ConfusionMatrix, th)
    return C.tp / C.pp
end   

function constraint(::Type{PrecisionGvRecall}, C::ConfusionMatrix, th)
    return C.tp / C.ap >= th
end   

precision_gv_recall_80 = PrecisionGvRecall(0.8)
special_case_positive!(precision_gv_recall_80)
cs_special_case_positive!(precision_gv_recall_80, true)

precision_gv_recall_60 = PrecisionGvRecall(0.6)
special_case_positive!(precision_gv_recall_60)
cs_special_case_positive!(precision_gv_recall_60, true)

precision_gv_recall_95 = PrecisionGvRecall(0.95)
special_case_positive!(precision_gv_recall_95)
cs_special_case_positive!(precision_gv_recall_95, true)


# recall given precision
@metric RecallGvPrecision th
function define(::Type{RecallGvPrecision}, C::ConfusionMatrix, th)
    return C.tp / C.pp
end   

function constraint(::Type{RecallGvPrecision}, C::ConfusionMatrix, th)
    return C.tp / C.ap >= th
end   

recal_gv_precision_80 = RecallGvPrecision(0.8)
special_case_positive!(recal_gv_precision_80)
cs_special_case_positive!(recal_gv_precision_80, true)


@metric PrecisionGvRecallSpecificity th1 th2        # precision given recall >= th1 and specificity >= th2
function define(::Type{PrecisionGvRecallSpecificity}, C::ConfusionMatrix, th1, th2)
    return C.tp / C.pp
end   
function constraint(::Type{PrecisionGvRecallSpecificity}, C::ConfusionMatrix, th1, th2)
    return [C.tp / C.ap >= th1,
            C.tn / C.an >= th2]
end   

precision_gv_recall_spec = PrecisionGvRecallSpecificity(0.8, 0.8)
special_case_positive!(precision_gv_recall_spec)
cs_special_case_positive!(precision_gv_recall_spec, [true, false])
cs_special_case_negative!(precision_gv_recall_spec, [false, true])

