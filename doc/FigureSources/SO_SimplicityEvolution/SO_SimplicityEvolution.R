simplicity_104 = read.table("simplicity_104", col.names = c("Kappa", "Simplicity"))
simplicity_107 = read.table("simplicity_107", col.names = c("Kappa", "Simplicity"))
simplicity_110 = read.table("simplicity_110", col.names = c("Kappa", "Simplicity"))

min_simplicity_104 = which.min(simplicity_104$Kappa)
min_simplicity_107 = which.min(simplicity_107$Kappa)
min_simplicity_110 = which.min(simplicity_110$Kappa)

generations_15 = seq(0, dim(simplicity_104)[1]-1)

plot(x = generations_15, y = simplicity_104$Kappa, type = "l", main = "Evolution of the average Kappa loss using simplicity",
     xlab = "Generations", ylab = "1 - Mean Kappa value",
     xlim = c(generations_15[1], generations_15[length(generations_15)]), 
     ylim = c(0,0.5), lwd = 2, col = "red")
lines(x = generations_15, y = simplicity_107$Kappa, lwd = 2, col = "blue")
lines(x = generations_15, y = simplicity_110$Kappa, lwd = 2, col = "green")
points(min_simplicity_104-1, simplicity_104$Kappa[min_simplicity_104], pch = 20)
points(min_simplicity_107-1, simplicity_107$Kappa[min_simplicity_107], pch = 20)
points(min_simplicity_110-1, simplicity_110$Kappa[min_simplicity_110], pch = 20)
legend(9, 0.48, 
       legend = c("Test subject 104","Test subject 107","Test subject 110"),
       col = c("red","blue","green"), lty = 1, lwd = 2, cex = 0.9)