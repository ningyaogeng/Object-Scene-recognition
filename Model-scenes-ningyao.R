# Code from Professor Josh de Leeuw
# I adapted this to analyze object data and for graphing 

# Although the minimal impact of exemplars may suggest
# that semantic structure is largely irrelevant, previous work has
# shown that abstract images that do not relate to preexisting
# knowledge are remembered very poorly (Koutstaal et al.,
# 2003; Wiseman & Neisser, 1974). Furthermore, in this study,
# we observed an 8% impairment in performance by adding 60
# additional exemplars within the same category to the study
# stream. By contrast, in the work of Standing (1973), an 8%
# drop in performance resulted from the addition of nearly 7,500
# categorically distinct items. Thus, visual categories may actually 
# provide the conceptual scaffolding for supporting detailed
# visual long-term memory for scenes, just as they do for objects
# (Konkle et al., 2010)

setwd(dir = '/Users/ningyaogeng/Desktop')

library(ggplot2)
library(ez)
library(dplyr)
library(tidyr)

# create MINERVA model

loss.parameter <- 0.1
tau.parameter <- 3 

num.features <- 8192

encode.item <- function(item, store, loss){
  lossy.item <- item * rbinom(length(item),1,prob = 1-loss) # What is lossy item?
  if(is.na(store)){
    store[1,] <- item
  } else {
    store <- rbind(store, item)
  }
}

# a is the trace and b is the probe. 
# The function computes the similarity of a particular trace to the probe.
# When the probe and trace are identical, returns a value of 1; 
# when they are orthogonoal, returns a value of 0. 

# Let a be the value of feature j in a trace, let b be the value of feature j in a probe.
# sum(a!=0|b!=0) is the number of relevant features in either the probe or the trace

similarity <- function(a, b){
  return(sum(a*b) / sum(a!=0|b!=0))
}


# Activation of a trace is the amount of similarity raised to the power of tau. Make tau 
# an odd number to preserve the sign. 

activation.level <- function(a, b, tau){
  return(similarity(a,b)^tau)
}


# The content of an echo is activation pattern across features. 
# The intensity is the summation of the activation levels of all traces. 
# It can be thought of as a measure of familiarity and it is used for modeling recognition judgements

echo.intensity <- function(item, store, tau){
  activations <- apply(store, 1, function(x){ return(activation.level(x, item, tau))})
  return(sum(activations))
}


echo.content <- function(item, store){
  activations <- apply(store, 1, function(x){ return(activation.level(x, item, tau))})
  return(sum(activations*store))
}

# try to predict results from Konkle, Brady, et al. study

# we have a total of 48 categories worth of features. 68 images per category.
# 3,264 total images.

# observers saw either 4, 16, or 64 items from each category
# 16 scene categories used for the test (4 in each condition).
# 4 items in each category, total of 64 tests.

# Other items were studied too, but maybe we can skip that detail for now?

# STUDY PHASE
# So we will have the model study 16 categories
# 1, 4, 16, or 64 per category

#test code
f <- table(features$category)

# Studied is 63 numbers/categories (twenty-one 4, 16, 64) in random order
# Studied.v is for each number x in studied, return True x times and False 68-x times
# Basically 63*68 numbers of TRUE/FALSE values

studied <- sample(rep(c(4,16,64), 21)) 
studied.v <- as.vector(sapply(studied, function(x){
  return(c(
    rep(T, x),
    rep(F, 68-x)
  ))
}))

# Assigning true/false values to randomly ordered features

features.random.order <- features %>% 
  group_by(category) %>%
  sample_frac(1) %>%
  ungroup() %>%
  mutate(studied = studied.v)

# ENCODE ALL STUDIED ITEMS INTO MODEL

studied.items <- features.random.order %>% filter(studied==T)

store <- matrix(NA, nrow=1, ncol=num.features)

for(i in 1:nrow(studied.items)){
  store <- encode.item(studied.items[[i,'features']], store, loss.parameter)
}

# TEST PHASE

# pick 4 studied, 4 unstudied items in each category to test

test.items <- features.random.order %>%
  group_by(category, studied) %>%
  sample_n(4) %>%
  mutate(item.num = 1:4)

echo.values <- c()
for(i in 1:nrow(test.items)){
  echo.values[i] <- echo.intensity(test.items[[i,'features']], store, tau.parameter)
}

test.results <- test.items %>%
  ungroup() %>%
  select(-path,-features) %>%
  mutate(echo.intensity = echo.values)

two.afc.scenes <- test.results %>%
  spread(studied,echo.intensity)%>%
  mutate(correct = `TRUE` > `FALSE`)

categories.conditions <- studied.items %>%
  group_by(category) %>%
  summarize(N=n())

scenes.summary <- two.afc.scenes %>% 
  left_join(categories.conditions) %>%
  group_by(N) %>%
  summarize(p.correct = mean(correct))


save(scenes.summary, file="scene_summary.Rdata")

# Graphing 

# Load the accuracy level from feature vectors extracted from the convolutional base of the
# VGG 16 CNN

load('scene_summary.Rdata')
load('object_summary.Rdata')

scenes.summary.conbase <- scenes.summary %>%
  mutate('type'= 'conbase') 

object.summary.conbase <- object.summary %>%
  mutate('type'= 'conbase') 

# Load the accuracy level from feature vectors extracted from the fourth pooled layer of the CNN

load('scene_summary_block4_pool.Rdata')
load('object_summary_block4_pool.Rdata')

scenes.summary.4 <- scenes.summary %>%
  mutate('type'= 'layer 4') 

object.summary.4 <- object.summary %>%
  mutate('type'= 'layer 4') 

# Load the accuracy level from feature vectors extracted from the second pooled layer of the CNN 

load('object_summary_block2_pool.Rdata')
load('scene_summary_block2_pool.Rdata')

scenes.summary.2 <- scenes.summary.block2 %>%
  mutate('type'= 'layer 2') 

object.summary.2 <- object.summary %>%
  mutate('type'= 'layer 2') 


human.scene.data <- data.frame(N=c(4, 16, 64), p.correct=c(0.84, 0.80,0.76), type='human')
scenes.summary <- bind_rows(scenes.summary.conbase,scenes.summary.4,scenes.summary.2, human.scene.data) 

human.object.data <- data.frame(N=c(1, 2, 4, 8, 16), p.correct=c(0.89, 0.88, 0.86, 0.82, 0.82), type='human')
object.summary <- bind_rows(object.summary.conbase,object.summary.4,object.summary.2, human.object.data) 

ggplot() + 
  geom_line(data = scenes.summary, aes(x = N, y = p.correct, color=type))+
  geom_point(data=scenes.summary,aes(x = N, y = p.correct, color=type), shape=0)+
  scale_y_continuous("Percentage Correct",limits=c(0.4, 1))+
  scale_x_sqrt("Number of Exemplars")+
  ggtitle("Scene recognition Memory comparison")

ggplot() + 
  geom_line(data = object.summary, aes(x = N, y = p.correct, color=type))+
  geom_point(data = object.summary,aes(x = N, y = p.correct, color=type), shape=0)+
  scale_y_continuous("Percentage Correct",limits=c(0.4, 1))+
  scale_x_sqrt("Number of Exemplars")+
  ggtitle("Object recognition Memory comparison")

scenes.summary.compare <- scenes.summary %>%
  filter(type!='layer 4'& type!='layer 2')

object.summary.compare <- object.summary %>%
  filter(type!='layer 4'& type!='layer 2')

ggplot() + 
  geom_line(data = scenes.summary, aes(x = N, y = p.correct, color=type))+
  # geom_line(data = object.summary, aes(x = N, y = p.correct, color=type))+
  geom_point(data=scenes.summary,aes(x = N, y = p.correct,color=type))+
  # geom_point(data = object.summary,aes(x = N, y = p.correct,color=type))+
  scale_y_continuous("Percentage Correct",limits=c(0.4, 1))+
  scale_x_sqrt("Number of Exemplars")+
  ggtitle("Scene Memory Recognition")

