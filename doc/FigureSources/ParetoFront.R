library("ggplot2")
x = c(18,12,25,15,22,20,10,30,15,30)
y = c(22,20,25,15,18,12,30,10,30,15)
front = c(7,2,4,6,8)
points = data.frame(x = x, y = y)
pts = data.frame(x = runif(50,10,30), y = runif(50,10,30))
pts = pts[pts$y > approx(points$x[front], points$y[front], xout = pts$x)$y,]
points = rbind(points,pts)
ggplot(data = points, aes(x = x, y = y, group = 1)) +
  geom_smooth(data = points[front, c("x","y")], color = "orange", size = 1.2) + 
  geom_point(data = points[front, c("x","y")], color = "blue", size = 2) +
  geom_point(data = points[-front, c("x","y")], size = 1.4) + theme_bw() +
  labs(x = expression(f[1](x)), y = expression(f[2](x))) +
  coord_fixed(ratio = 1) +
  theme(panel.border = element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        axis.text.x = element_blank(), axis.ticks.x = element_blank(),
        axis.text.y = element_blank(), axis.ticks.y = element_blank(),
        axis.line = element_line(color = "black"))