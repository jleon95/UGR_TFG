i300_g150_104 = read.table("300i_150g_104", col.names = c("Kappa", "CV"))
i300_g150_107 = read.table("300i_150g_107", col.names = c("Kappa", "CV"))
i300_g150_110 = read.table("300i_150g_110", col.names = c("Kappa", "CV"))

i500_g100_104 = read.table("500i_100g_104", col.names = c("Kappa", "CV"))
i500_g100_107 = read.table("500i_100g_107", col.names = c("Kappa", "CV"))
i500_g100_110 = read.table("500i_100g_110", col.names = c("Kappa", "CV"))

i800_g200_104 = read.table("800i_200g_104", col.names = c("Kappa", "CV"))
i800_g200_107 = read.table("800i_200g_107", col.names = c("Kappa", "CV"))
i800_g200_110 = read.table("800i_200g_110", col.names = c("Kappa", "CV"))

generations_150 = seq(0, dim(i300_g150_104)[1]-1)
generations_100 = seq(0, dim(i500_g100_104)[1]-1)
generations_200 = seq(0, dim(i800_g200_104)[1]-1)