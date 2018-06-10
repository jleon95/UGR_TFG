i300_g150_104 = unlist(read.table("300i_150g_104"), use.names = FALSE)
i300_g150_107 = unlist(read.table("300i_150g_107"), use.names = FALSE)
i300_g150_110 = unlist(read.table("300i_150g_110"), use.names = FALSE)

i500_g100_104 = unlist(read.table("500i_100g_104"), use.names = FALSE)
i500_g100_107 = unlist(read.table("500i_100g_107"), use.names = FALSE)
i500_g100_110 = unlist(read.table("500i_100g_110"), use.names = FALSE)

i800_g200_104 = unlist(read.table("800i_200g_104"), use.names = FALSE)
i800_g200_107 = unlist(read.table("800i_200g_107"), use.names = FALSE)
i800_g200_110 = unlist(read.table("800i_200g_110"), use.names = FALSE)

barplot(i300_g150_104, ylim = c(0,300))
barplot(i500_g100_104, ylim = c(0,500))
barplot(i800_g200_104, ylim = c(0,800))

barplot(sqrt(table(i300_g150_104)), col = "royalblue", cex.names = 0.95)
barplot(sqrt(table(i500_g100_104)), col = "royalblue", cex.names = 0.95)
barplot(sqrt(table(i800_g200_104)), col = "royalblue", cex.names = 0.95)