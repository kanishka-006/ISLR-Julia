using Random
using Plots
using CSV
using DataFrames
using MLJ
using MLJLIBSVMInterface
using CategoricalArrays
using ROC
import LIBSVM
import Statistics
using PrettyPrinting

function plot_svc(model, x, y, count,h=0.02, pad=0.25)
    x_min = minimum(x[:, 1]) - pad
    x_max = maximum(x[:, 1]) + pad
    y_min = minimum(x[:, 2]) - pad
    y_max = maximum(x[:, 2]) + pad
    
    xx = x_min:h:x_max
    yy = y_min:h:y_max
    
    f(i, j) = begin 
        c = reshape([i; j], (2, 1))
        pred = svmpredict(model, c) 
        return pred[2][1]
    end
    

    contour(xx, yy, f, fill = true, colorbar_entry=false, levels=count)
end