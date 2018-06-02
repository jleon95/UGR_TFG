i300_g150_104 = read.table("300i_150g_104", col.names = c("Kappa", "CV"))
i300_g150_107 = read.table("300i_150g_107", col.names = c("Kappa", "CV"))
i300_g150_110 = read.table("300i_150g_110", col.names = c("Kappa", "CV"))

i500_g100_104 = read.table("500i_100g_104", col.names = c("Kappa", "CV"))
i500_g100_107 = read.table("500i_100g_107", col.names = c("Kappa", "CV"))
i500_g100_110 = read.table("500i_100g_110", col.names = c("Kappa", "CV"))

i800_g200_104 = read.table("800i_200g_104", col.names = c("Kappa", "CV"))
i800_g200_107 = read.table("800i_200g_107", col.names = c("Kappa", "CV"))
i800_g200_110 = read.table("800i_200g_110", col.names = c("Kappa", "CV"))

generations_150 = seq(0, dim(i300_g150_104)[1]-1)
generations_100 = seq(0, dim(i500_g100_104)[1]-1)
generations_200 = seq(0, dim(i800_g200_104)[1]-1)

# Kappa and CV individuals and generations comparison for subject 104
plot(x = generations_200, y = i800_g200_104$Kappa, type = "l", main = "104: individuals and generations comparison (Kappa)",
     xlab = "Generations", ylab = "1 - Mean Kappa value",
     xlim = c(generations_200[1], generations_200[length(generations_200)]), 
     ylim = c(0,1), lwd = 2, col = "blue")
lines(x = generations_150, y = i300_g150_104$Kappa, lwd = 2, col = "red")
lines(x = generations_100, y = i500_g100_104$Kappa, lwd = 2, col = "green")
legend(85, 0.99, 
       legend = c("800 individuals, 200 generations","300 individuals, 150 generations","500 individuals, 100 generations"),
       col = c("blue","red","green"), lty = 1, lwd = 2, cex = 0.9)

plot(x = generations_200, y = i800_g200_104$CV, type = "l", main = "104: individuals and generations comparison (CV)",
     xlab = "Generations", ylab = "Mean cross validation error",
     xlim = c(generations_200[1], generations_200[length(generations_200)]), 
     ylim = c(0,1), lwd = 2, col = "blue")
lines(x = generations_150, y = i300_g150_104$CV, lwd = 2, col = "red")
lines(x = generations_100, y = i500_g100_104$CV, lwd = 2, col = "green")
legend(85, 0.99, 
       legend = c("800 individuals, 200 generations","300 individuals, 150 generations","500 individuals, 100 generations"),
       col = c("blue","red","green"), lty = 1, lwd = 2, cex = 0.9)

# Kappa and CV individuals and generations comparison for subject 107
plot(x = generations_200, y = i800_g200_107$Kappa, type = "l", main = "107: individuals and generations comparison (Kappa)",
     xlab = "Generations", ylab = "1 - Mean Kappa value",
     xlim = c(generations_200[1], generations_200[length(generations_200)]), 
     ylim = c(0,1), lwd = 2, col = "blue")
lines(x = generations_150, y = i300_g150_107$Kappa, lwd = 2, col = "red")
lines(x = generations_100, y = i500_g100_107$Kappa, lwd = 2, col = "green")
legend(85, 0.99, 
       legend = c("800 individuals, 200 generations","300 individuals, 150 generations","500 individuals, 100 generations"),
       col = c("blue","red","green"), lty = 1, lwd = 2, cex = 0.9)

plot(x = generations_200, y = i800_g200_107$CV, type = "l", main = "107: individuals and generations comparison (CV)",
     xlab = "Generations", ylab = "Mean cross validation error",
     xlim = c(generations_200[1], generations_200[length(generations_200)]), 
     ylim = c(0,1), lwd = 2, col = "blue")
lines(x = generations_150, y = i300_g150_107$CV, lwd = 2, col = "red")
lines(x = generations_100, y = i500_g100_107$CV, lwd = 2, col = "green")
legend(85, 0.99, 
       legend = c("800 individuals, 200 generations","300 individuals, 150 generations","500 individuals, 100 generations"),
       col = c("blue","red","green"), lty = 1, lwd = 2, cex = 0.9)

# Kappa and CV individuals and generations comparison for subject 110
plot(x = generations_200, y = i800_g200_110$Kappa, type = "l", main = "110: individuals and generations comparison (Kappa)",
     xlab = "Generations", ylab = "1 - Mean Kappa value",
     xlim = c(generations_200[1], generations_200[length(generations_200)]), 
     ylim = c(0,1), lwd = 2, col = "blue")
lines(x = generations_150, y = i300_g150_110$Kappa, lwd = 2, col = "red")
lines(x = generations_100, y = i500_g100_110$Kappa, lwd = 2, col = "green")
legend(85, 0.99, 
       legend = c("800 individuals, 200 generations","300 individuals, 150 generations","500 individuals, 100 generations"),
       col = c("blue","red","green"), lty = 1, lwd = 2, cex = 0.9)

plot(x = generations_200, y = i800_g200_110$CV, type = "l", main = "110: individuals and generations comparison (CV)",
     xlab = "Generations", ylab = "Mean cross validation error",
     xlim = c(generations_200[1], generations_200[length(generations_200)]), 
     ylim = c(0,1), lwd = 2, col = "blue")
lines(x = generations_150, y = i300_g150_110$CV, lwd = 2, col = "red")
lines(x = generations_100, y = i500_g100_110$CV, lwd = 2, col = "green")
legend(85, 0.99, 
       legend = c("800 individuals, 200 generations","300 individuals, 150 generations","500 individuals, 100 generations"),
       col = c("blue","red","green"), lty = 1, lwd = 2, cex = 0.9)