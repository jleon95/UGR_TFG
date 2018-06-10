simplicity_104 = read.table("simplicity_104", col.names = c("Kappa", "CV", "Simplicity"))
simplicity_107 = read.table("simplicity_107", col.names = c("Kappa", "CV", "Simplicity"))
simplicity_110 = read.table("simplicity_110", col.names = c("Kappa", "CV", "Simplicity"))

not_simplicity_104 = read.table("not_simplicity_104", col.names = c("Kappa", "CV"))
not_simplicity_107 = read.table("not_simplicity_107", col.names = c("Kappa", "CV"))
not_simplicity_110 = read.table("not_simplicity_110", col.names = c("Kappa", "CV"))

generations_200 = seq(0, dim(simplicity_104)[1]-1)

# Kappa comparison for simplicity and not simplicity fitness in subject 104
plot(x = generations_200, y = simplicity_104$Kappa, type = "l", main = "104: simplicity use (Kappa)",
     xlab = "Generations", ylab = "1 - Mean Kappa value",
     xlim = c(generations_200[1], generations_200[length(generations_200)]), 
     ylim = c(0,1), lwd = 2, col = "red")
lines(x = generations_200, y = not_simplicity_104$Kappa, lwd = 2, col = "blue")
legend(85, 0.96, 
       legend = c("Using the simplicity measure","Not using the simplicity measure"),
       col = c("red","blue"), lty = 1, lwd = 2, cex = 0.9)

# Kappa comparison for simplicity and not simplicity fitness in subject 107
plot(x = generations_200, y = simplicity_107$Kappa, type = "l", main = "107: simplicity use (Kappa)",
     xlab = "Generations", ylab = "1 - Mean Kappa value",
     xlim = c(generations_200[1], generations_200[length(generations_200)]), 
     ylim = c(0,1), lwd = 2, col = "red")
lines(x = generations_200, y = not_simplicity_107$Kappa, lwd = 2, col = "blue")
legend(85, 0.96, 
       legend = c("Using the simplicity measure","Not using the simplicity measure"),
       col = c("red","blue"), lty = 1, lwd = 2, cex = 0.9)

# Kappa comparison for simplicity and not simplicity fitness in subject 110
plot(x = generations_200, y = simplicity_110$Kappa, type = "l", main = "110: simplicity use (Kappa)",
     xlab = "Generations", ylab = "1 - Mean Kappa value",
     xlim = c(generations_200[1], generations_200[length(generations_200)]), 
     ylim = c(0,1), lwd = 2, col = "red")
lines(x = generations_200, y = not_simplicity_110$Kappa, lwd = 2, col = "blue")
legend(85, 0.96, 
       legend = c("Using the simplicity measure","Not using the simplicity measure"),
       col = c("red","blue"), lty = 1, lwd = 2, cex = 0.9)