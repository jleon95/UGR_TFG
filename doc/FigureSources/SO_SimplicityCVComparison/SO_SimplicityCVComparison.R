simplicity_104 = read.table("simplicity_104", col.names = c("Kappa", "Simplicity"))
simplicity_107 = read.table("simplicity_107", col.names = c("Kappa", "Simplicity"))
simplicity_110 = read.table("simplicity_110", col.names = c("Kappa", "Simplicity"))

min_simplicity_104 = which.min(simplicity_104$Kappa)
min_simplicity_107 = which.min(simplicity_107$Kappa)
min_simplicity_110 = which.min(simplicity_110$Kappa)

cv_104 = read.table("cv_104", col.names = c("Kappa", "CV"))
cv_107 = read.table("cv_107", col.names = c("Kappa", "CV"))
cv_110 = read.table("cv_110", col.names = c("Kappa", "CV"))

min_cv_104 = which.min(cv_104$Kappa)
min_cv_107 = which.min(cv_107$Kappa)
min_cv_110 = which.min(cv_110$Kappa)

generations_20 = seq(0, dim(simplicity_104)[1]-1)

# Kappa comparison for simplicity and cross-validation in subject 104
plot(x = generations_20, y = simplicity_104$Kappa, type = "l", main = "104: simplicity use (Kappa)",
     xlab = "Generations", ylab = "1 - Mean Kappa value",
     xlim = c(generations_20[1], generations_20[length(generations_20)]), 
     ylim = c(0,1), lwd = 2, col = "red")
lines(x = generations_20, y = cv_104$Kappa, lwd = 2, col = "blue")
points(min_simplicity_104-1, simplicity_104$Kappa[min_simplicity_104], pch = 20)
points(min_cv_104-1, cv_104$Kappa[min_cv_104], pch = 20)
legend(12, 0.96, 
       legend = c("Simplicity","Cross-validation"),
       col = c("red","blue"), lty = 1, lwd = 2, cex = 0.9)

# Kappa comparison for simplicity and cross-validation in subject 107
plot(x = generations_20, y = simplicity_107$Kappa, type = "l", main = "107: simplicity use (Kappa)",
     xlab = "Generations", ylab = "1 - Mean Kappa value",
     xlim = c(generations_20[1], generations_20[length(generations_20)]), 
     ylim = c(0,1), lwd = 2, col = "red")
lines(x = generations_20, y = cv_107$Kappa, lwd = 2, col = "blue")
points(min_simplicity_107-1, simplicity_107$Kappa[min_simplicity_107], pch = 20)
points(min_cv_107-1, cv_107$Kappa[min_cv_107], pch = 20)
legend(12, 0.96, 
       legend = c("Simplicity","Cross-validation"),
       col = c("red","blue"), lty = 1, lwd = 2, cex = 0.9)

# Kappa comparison for simplicity and cross-validation in subject 110
plot(x = generations_20, y = simplicity_110$Kappa, type = "l", main = "110: simplicity use (Kappa)",
     xlab = "Generations", ylab = "1 - Mean Kappa value",
     xlim = c(generations_20[1], generations_20[length(generations_20)]), 
     ylim = c(0,1), lwd = 2, col = "red")
lines(x = generations_20, y = cv_110$Kappa, lwd = 2, col = "blue")
points(min_simplicity_110-1, simplicity_110$Kappa[min_simplicity_110], pch = 20)
points(min_cv_110-1, cv_110$Kappa[min_cv_110], pch = 20)
legend(12, 0.96, 
       legend = c("Simplicity","Cross-validation"),
       col = c("red","blue"), lty = 1, lwd = 2, cex = 0.9)