parallel_fs = read.table("fs_parallel_incremental.txt", col.names = c("Threads", "Time"))

plot(x = parallel_fs$Threads, y = parallel_fs$Time, type = "l", main = "Evolution of feature selection time with multiple threads",
     xlab = "Number of threads", ylab = "Time (s)",
     xlim = c(parallel_fs$Threads[1]-1,parallel_fs$Threads[length(parallel_fs$Threads)]), 
     ylim = c(0,400), lwd = 2, col = "blue", xaxt = "n")
axis(1, at = c(1,2,4,6,8,10,12,14,16), labels = c(1,2,4,6,8,10,12,14,16))
points(x = 1, y = 254.792, pch = 20, col = "red")
legend(11, 395, 
       legend = c("Single thread","Multiple threads"),
       col = c("red","blue"), lty = c(NA,1), lwd = c(NA,2), pch = c(20,NA), cex = 0.9)
