library(ggplot2)
library(dplyr)
library(stringr)
library(ggpmisc)

v0_data <- read.csv("v0_data.csv")
avsr_data <- read.csv("avsr_data.csv")
sst_data <- read.csv("stt_data.csv")

names(v0_data) <- c("Trial", "Duration", "Latency", "Correct", "Total", "WER", 
                    "Noise", "Dim")

names(avsr_data) <- c("Trial", "Duration", "Latency", "Correct", "Total", "WER", 
                      "SNR", "Noise", "Dim", "Topic")

names(sst_data) <- c("Trial", "Duration", "Correct", "Total", "WER", 
                     "SNR", "Noise", "Topic")
sst_data <- sst_data %>% select(-c(Duration))
sst_data$Noise <- factor(sst_data$Noise, levels=c('White Noise', "Music", "Speaking", "None"))


avsr_data$WER <- 100*(avsr_data$WER)
avsr_data <- avsr_data[-c(2),]
avsr_data <- avsr_data %>% mutate(Noise = str_replace(Noise, "N/A", "None"))
avsr_data$Noise <- factor(avsr_data$Noise, levels=c('White Noise', "Music", "Speaking", "None"))


# 95% confidence interval loess regression
ggplot(avsr_data, aes(x=Duration, y=WER)) +
  geom_jitter(aes(shape=Noise), size=2.5) +
  geom_smooth(method="auto", se=TRUE, aes(color="Loess Regression")) +
  ggtitle("Word Error Rate (WER) vs Video Duration") +
  labs(x="Duration (s)", y="WER (%)", color=NULL, shape="Noise Type") +
  scale_x_continuous(breaks = seq(0, 30, by = 5)) +
  theme_classic() +
  theme(text=element_text(family="Helvetica", size=18)) 

# ggplot(avsr_data, aes(x=Duration, y=WER)) +
#   geom_histogram(stat="identity")

ggsave("wervsduration.png", plot=last_plot(), device="png")

v0_latency <- v0_data[,c("Duration", "Latency")]
v0_latency$System <- "Unoptimized"
v1_latency <- avsr_data[,c("Duration", "Latency")]
v1_latency$System <- "Optimized"

combined_latency <- rbind(v0_latency, v1_latency)

# ggplot(avsr_data, aes(x=Duration, y=Latency)) +
#   geom_jitter(size=2,shape="circle") +
#   geom_smooth(method="lm", se=TRUE, aes(color="Optimized System")) +
#   ggtitle("Transcription Latency vs Video Duration") +
#   labs(x="Duration (s)", y="Latency (s)", color=NULL)

ggplot(combined_latency, aes(x=Duration, y=Latency)) +
  geom_jitter(size=2.5, aes(shape=System)) +
  geom_smooth(method="lm", se=TRUE, aes(color=System)) +
  ggtitle("Transcription Latency vs Video Duration") +
  labs(x="Duration (s)", y="Latency (s)", color=NULL) +
  scale_x_continuous(breaks = seq(0, 30, by = 5)) +
  theme_classic() +
  theme(text=element_text(family="Helvetica", size=18))

ggsave("latencyvsduration.png", plot=last_plot(), device="png")

fitted_models <- combined_latency %>% group_by(System) %>% do(model = lm(Latency ~ Duration, data = .))
fitted_models$model

# Optimized: 1.025d + 3.881
# Unoptimized: 6.038d + 6.219


avsr_updated <- avsr_data %>% filter(Duration <= 15)

wer_rates_noise_avsr <- avsr_updated %>% group_by(Noise, SNR) %>% 
  summarize(WER=(1-sum(Correct)/sum(Total))*100)
wer_rates_noise_avsr

wer_rates_noise_sst <- sst_data %>% group_by(Noise, SNR) %>% 
  summarize(WER=(1-sum(Correct)/sum(Total))*100)
wer_rates_noise_sst



wer_rates_topic_avsr <- avsr_updated %>% group_by(Topic, SNR) %>% 
  summarize(WER=(1-sum(Correct)/sum(Total))*100)
wer_rates_topic_avsr

wer_rates_topic_sst <- sst_data %>% group_by(Topic, SNR) %>% 
  summarize(WER=(1-sum(Correct)/sum(Total))*100)
wer_rates_topic_sst



wer_rates_dim <- avsr_updated %>% group_by(Dim, SNR) %>% 
  summarize(WER=(1-sum(Correct)/sum(Total))*100)
wer_rates_dim

sum(avsr_updated$Correct)
sum(avsr_updated$Total)
sum(sst_data$Correct)
sum(sst_data$Total)

prop.test(x=c(2632, 3414), n=c(2730, 4200))

group1 <- avsr_updated %>% group_by(Topic, SNR) %>% 
  summarize(tot_correct=sum(Correct), tot_tot = sum(Total))

group2 <- sst_data %>% group_by(Topic, SNR) %>% 
  summarize(tot_correct=sum(Correct), tot_tot = sum(Total))

for (i in 1:length(group1$Topic))
{
  print(group1[i,])
  print(prop.test(x=c(group1$tot_correct[i], group2$tot_correct[i]),
            n=c(group1$tot_tot[i], group2$tot_tot[i])))
}



g1 <- avsr_updated %>% group_by(Noise, SNR) %>% 
  summarize(tot_correct=sum(Correct), tot_tot = sum(Total))

g2 <- sst_data %>% group_by(Noise, SNR) %>% 
  summarize(tot_correct=sum(Correct), tot_tot = sum(Total))

for (i in 1:length(g1$Noise))
{
  print(g1[i,])
  print(prop.test(x=c(g1$tot_correct[i], g2$tot_correct[i]),
                  n=c(g1$tot_tot[i], g2$tot_tot[i])))
}




