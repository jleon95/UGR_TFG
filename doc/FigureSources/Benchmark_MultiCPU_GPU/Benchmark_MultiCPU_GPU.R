multicpu = read.table("multi_cpu-all.txt", col.names = c("Threads", "Time"))

plot(x = multicpu$Threads, y = multicpu$Time, type = "l", main = "Evolution of training time with multiple threads",
     xlab = "Number of threads", ylab = "Time (s)",
     xlim = c(multicpu$Threads[1]-1,multicpu$Threads[length(multicpu$Threads)]), 
     ylim = c(0,2500), lwd = 2, col = "blue", xaxt = "n")
axis(1, at = c(1,4,8,12,16,20,24), labels = c(1,4,8,12,16,20,24))
points(x = 1, y = 2430.4218711853027, pch = 20, col = "red")
legend(16, 2450, 
       legend = c("GPU","Multiple threads"),
       col = c("red","blue"), lty = c(NA,1), lwd = c(NA,2), pch = c(20,NA), cex = 0.9)
