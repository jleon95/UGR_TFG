cpu = read.table("cpu", col.names = c("Time", "Neurons"))
gpu = read.table("gpu", col.names = c("Time", "Neurons"))

plot(x = cpu$Neurons, y = cpu$Time, type = "l", main = "CPU vs. GPU computational time",
     xlab = "Units in 1-layer neural network", ylab = "Time (s)",
     xlim = c(cpu$Neurons[1], cpu$Neurons[length(cpu$Neurons)]), 
     ylim = c(0,10), lwd = 2, col = "red", xaxt = "n")
options(scipen = 5)
axis(1, at = c(4400, 20000, 40000, 60000, 80000, 100000), labels = c(4400, 20000, 40000, 60000, 80000, 100000))
lines(x = gpu$Neurons, y = gpu$Time, lwd = 2, col = "blue")
points(x = 4400, y = 0.9, pch = 20)