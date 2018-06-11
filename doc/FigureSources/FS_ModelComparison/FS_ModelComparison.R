logreg_104 = read.table("logreg_104", col.names = c("Kappa", "CV"))
logreg_107 = read.table("logreg_107", col.names = c("Kappa", "CV"))
logreg_110 = read.table("logreg_110", col.names = c("Kappa", "CV"))

svm_104 = read.table("svm_104", col.names = c("Kappa", "CV"))
svm_107 = read.table("svm_107", col.names = c("Kappa", "CV"))
svm_110 = read.table("svm_110", col.names = c("Kappa", "CV"))

generations_200 = seq(0, dim(svm_104)[1]-1)

# Kappa and CV comparison for LogReg and SVM (subject 104)
plot(x = generations_200, y = logreg_104$Kappa, type = "l", main = "104: Model comparison (Kappa)",
     xlab = "Generations", ylab = "1 - Mean Kappa value",
     xlim = c(generations_200[1], generations_200[length(generations_200)]), 
     ylim = c(0,1), lwd = 2, col = "red")
lines(x = generations_200, y = svm_104$Kappa, lwd = 2, col = "blue")
legend(115, 0.97, 
       legend = c("Logistic Regression","SVM"),
       col = c("red","blue"), lty = 1, lwd = 2, cex = 0.9)

plot(x = generations_200, y = logreg_104$CV, type = "l", main = "104: Model comparison (CV)",
     xlab = "Generations", ylab = "Mean cross-validation error",
     xlim = c(generations_200[1], generations_200[length(generations_200)]), 
     ylim = c(0,1), lwd = 2, col = "red")
lines(x = generations_200, y = svm_104$CV, lwd = 2, col = "blue")
legend(115, 0.97, 
       legend = c("Logistic Regression","SVM"),
       col = c("red","blue"), lty = 1, lwd = 2, cex = 0.9)

# Kappa and CV comparison for LogReg and SVM (subject 107)
plot(x = generations_200, y = logreg_107$Kappa, type = "l", main = "107: Model comparison (Kappa)",
     xlab = "Generations", ylab = "1 - Mean Kappa value",
     xlim = c(generations_200[1], generations_200[length(generations_200)]), 
     ylim = c(0,1), lwd = 2, col = "red")
lines(x = generations_200, y = svm_107$Kappa, lwd = 2, col = "blue")
legend(115, 0.97, 
       legend = c("Logistic Regression","SVM"),
       col = c("red","blue"), lty = 1, lwd = 2, cex = 0.9)

plot(x = generations_200, y = logreg_107$CV, type = "l", main = "107: Model comparison (CV)",
     xlab = "Generations", ylab = "Mean cross-validation error",
     xlim = c(generations_200[1], generations_200[length(generations_200)]), 
     ylim = c(0,1), lwd = 2, col = "red")
lines(x = generations_200, y = svm_107$CV, lwd = 2, col = "blue")
legend(115, 0.97, 
       legend = c("Logistic Regression","SVM"),
       col = c("red","blue"), lty = 1, lwd = 2, cex = 0.9)

# Kappa and CV comparison for LogReg and SVM (subject 110)
plot(x = generations_200, y = logreg_110$Kappa, type = "l", main = "110: Model comparison (Kappa)",
     xlab = "Generations", ylab = "1 - Mean Kappa value",
     xlim = c(generations_200[1], generations_200[length(generations_200)]), 
     ylim = c(0,1), lwd = 2, col = "red")
lines(x = generations_200, y = svm_110$Kappa, lwd = 2, col = "blue")
legend(115, 0.97, 
       legend = c("Logistic Regression","SVM"),
       col = c("red","blue"), lty = 1, lwd = 2, cex = 0.9)

plot(x = generations_200, y = logreg_110$CV, type = "l", main = "110: Model comparison (CV)",
     xlab = "Generations", ylab = "Mean cross-validation error",
     xlim = c(generations_200[1], generations_200[length(generations_200)]), 
     ylim = c(0,1), lwd = 2, col = "red")
lines(x = generations_200, y = svm_110$CV, lwd = 2, col = "blue")
legend(115, 0.97, 
       legend = c("Logistic Regression","SVM"),
       col = c("red","blue"), lty = 1, lwd = 2, cex = 0.9)