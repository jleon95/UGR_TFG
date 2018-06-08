uniform_104 = read.table("uniform_crossover_104", col.names = c("Kappa", "CV"))
uniform_107 = read.table("uniform_crossover_107", col.names = c("Kappa", "CV"))
uniform_110 = read.table("uniform_crossover_110", col.names = c("Kappa", "CV"))

singlepoint_104 = read.table("singlepoint_crossover_104", col.names = c("Kappa", "CV"))
singlepoint_107 = read.table("singlepoint_crossover_107", col.names = c("Kappa", "CV"))
singlepoint_110 = read.table("singlepoint_crossover_110", col.names = c("Kappa", "CV"))

twopoint_104 = read.table("twopoint_crossover_104", col.names = c("Kappa", "CV"))
twopoint_107 = read.table("twopoint_crossover_107", col.names = c("Kappa", "CV"))
twopoint_110 = read.table("twopoint_crossover_110", col.names = c("Kappa", "CV"))

generations = seq(0, dim(uniform_104)[1]-1)

# Kappa and CV comparison for subject 104
plot(x = generations, y = uniform_104$Kappa, type = "l", main = "104: Crossover comparison (Kappa)",
    xlab = "Generations", ylab = "1 - Mean Kappa value",
    xlim = c(generations[1], generations[length(generations)]), 
    ylim = c(0,1), lwd = 2, col = "blue")
lines(x = generations, y = singlepoint_104$Kappa, lwd = 2, col = "red")
lines(x = generations, y = twopoint_104$Kappa, lwd = 2, col = "green")
legend(85, 1, legend = c("Uniform crossover","Single-point crossover","Two-point crossover"),
      col = c("blue","red","green"), lty = 1, lwd = 2, cex = 0.9)

plot(x = generations, y = uniform_104$CV, type = "l", main = "104: Crossover comparison (CV)",
     xlab = "Generations", ylab = "Mean cross-validation error",
     xlim = c(generations[1], generations[length(generations)]), 
     ylim = c(0,1), lwd = 2, col = "blue")
lines(x = generations, y = singlepoint_104$CV, lwd = 2, col = "red")
lines(x = generations, y = twopoint_104$CV, lwd = 2, col = "green")
legend(85, 1, legend = c("Uniform crossover","Single-point crossover","Two-point crossover"),
       col = c("blue","red","green"), lty = 1, lwd = 2, cex = 0.9)

# Kappa and CV comparison for subject 107
plot(x = generations, y = uniform_107$Kappa, type = "l", main = "107: Crossover comparison (Kappa)",
     xlab = "Generations", ylab = "1 - Mean Kappa value",
     xlim = c(generations[1], generations[length(generations)]), 
     ylim = c(0,1), lwd = 2, col = "blue")
lines(x = generations, y = singlepoint_107$Kappa, lwd = 2, col = "red")
lines(x = generations, y = twopoint_107$Kappa, lwd = 2, col = "green")
legend(85, 1, legend = c("Uniform crossover","Single-point crossover","Two-point crossover"),
       col = c("blue","red","green"), lty = 1, lwd = 2, cex = 0.9)

plot(x = generations, y = uniform_107$CV, type = "l", main = "107: Crossover comparison (CV)",
     xlab = "Generations", ylab = "Mean cross-validation error",
     xlim = c(generations[1], generations[length(generations)]), 
     ylim = c(0,1), lwd = 2, col = "blue")
lines(x = generations, y = singlepoint_107$CV, lwd = 2, col = "red")
lines(x = generations, y = twopoint_107$CV, lwd = 2, col = "green")
legend(85, 1, legend = c("Uniform crossover","Single-point crossover","Two-point crossover"),
       col = c("blue","red","green"), lty = 1, lwd = 2, cex = 0.9)

# Kappa and CV comparison for subject 110
plot(x = generations, y = uniform_110$Kappa, type = "l", main = "110: Crossover comparison (Kappa)",
     xlab = "Generations", ylab = "1 - Mean Kappa value",
     xlim = c(generations[1], generations[length(generations)]), 
     ylim = c(0,1), lwd = 2, col = "blue")
lines(x = generations, y = singlepoint_110$Kappa, lwd = 2, col = "red")
lines(x = generations, y = twopoint_110$Kappa, lwd = 2, col = "green")
legend(85, 1, legend = c("Uniform crossover","Single-point crossover","Two-point crossover"),
       col = c("blue","red","green"), lty = 1, lwd = 2, cex = 0.9)

plot(x = generations, y = uniform_110$CV, type = "l", main = "110: Crossover comparison (CV)",
     xlab = "Generations", ylab = "Mean cross-validation error",
     xlim = c(generations[1], generations[length(generations)]), 
     ylim = c(0,1), lwd = 2, col = "blue")
lines(x = generations, y = singlepoint_110$CV, lwd = 2, col = "red")
lines(x = generations, y = twopoint_110$CV, lwd = 2, col = "green")
legend(85, 1, legend = c("Uniform crossover","Single-point crossover","Two-point crossover"),
       col = c("blue","red","green"), lty = 1, lwd = 2, cex = 0.9)
