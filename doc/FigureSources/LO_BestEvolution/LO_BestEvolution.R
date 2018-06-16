learning_104 = read.table("learning_104", col.names = c("Kappa", "CV"))
learning_107 = read.table("learning_107", col.names = c("Kappa", "CV"))
learning_110 = read.table("learning_110", col.names = c("Kappa", "CV"))

min_104 = which.min(learning_104$Kappa)
min_107 = which.min(learning_107$Kappa)
min_110 = which.min(learning_110$Kappa)

learning_20 = seq(0, dim(learning_104)[1]-1)

plot(x = generations_20, y = learning_104$Kappa, type = "l", main = "Evolution of the average Kappa loss in learning optimization",
     xlab = "Generations", ylab = "1 - Mean Kappa value",
     xlim = c(generations_20[1], generations_20[length(generations_20)]), 
     ylim = c(0,0.5), lwd = 2, col = "red")
lines(x = generations_20, y = learning_107$Kappa, lwd = 2, col = "blue")
lines(x = generations_20, y = learning_110$Kappa, lwd = 2, col = "green")
points(min_104-1, learning_104$Kappa[min_104], pch = 20)
points(min_107-1, learning_107$Kappa[min_107], pch = 20)
points(min_110-1, learning_110$Kappa[min_110], pch = 20)
legend(9, 0.48, 
       legend = c("Test subject 104","Test subject 107","Test subject 110"),
       col = c("red","blue","green"), lty = 1, lwd = 2, cex = 0.9)