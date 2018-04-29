valores = seq(-2, 2, by=0.02)
relu = function(x){max(0, x)}
leaky_relu = function(x){if(x>0)x else 0.1*x}
elu = function(x, a){if(x>0)x else a*(exp(x)-1)}
resultados_tanh = lapply(valores, tanh)
resultados_relu = lapply(valores, relu)
resultados_leaky_relu = lapply(valores, leaky_relu)
resultados_elu = lapply(valores, elu, a = 0.5)
par(mfrow = c(2,2))
par(pty = "s")
plot(x = valores, y = resultados_tanh, type = 'l',
     main = 'TanH', xlab = '', ylab = '', xlim = c(-2,2), ylim = c(-2,2),
     lwd = 2, col = 'blue', xaxt = 'n', yaxt = 'n')
abline(h = 0)
plot(x = valores, y = resultados_relu, type = 'l',
     main = 'ReLU', xlab = '', ylab = '', xlim = c(-2,2), ylim = c(-2,2),
     lwd = 2, col = 'blue', xaxt = 'n', yaxt = 'n')
abline(h = 0)
plot(x = valores, y = resultados_leaky_relu, type = 'l',
     main = 'Leaky ReLU', xlab = '', ylab = '', xlim = c(-2,2), ylim = c(-2,2), 
     lwd = 2, col = 'blue', xaxt = 'n', yaxt = 'n')
abline(h = 0)
plot(x = valores, y = resultados_elu, type = 'l',
     main = 'ELU', xlab = '', ylab = '', xlim = c(-2,2), ylim = c(-2,2), 
     lwd = 2, col = 'blue', xaxt = 'n', yaxt = 'n')
abline(h = 0)
par(mfrow = c(1,1))
par(pty = "m")