# Code from Professor Josh de Leeuw
# I adapted this to use different layers of the CNN to extract features
# I also adapted this for the processing of object data
# I also adapted this for RGB processing but I forgot to email it to myself before leaving so I left it out of the essay. 

# load all the images
# get VGG feature representations of them

library(keras)
library(dplyr)
library(stringr)
library(tensorflow)

setwd(dir = '/Users/research/Desktop')

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(128,128,3)
)

conv_base %>% summary()

extract_features <- function(directory, class_name){
  
  all.files <- list.files(path=directory, pattern=".jpg")
  num.files <- length(all.files)
  features <- array(0, dim=c(num.files, 4, 4, 512))
  inputs <- array(0, dim=c(num.files, 128, 128, 3))
  
  for(i in 1:num.files){
    img <- image_load(file.path(directory,all.files[i]), target_size = c(128,128)) %>%
      image_to_array() %>%
      array_reshape(c(1, dim(.))) %>%
      imagenet_preprocess_input(mode="tf")
    inputs[i,,,] <- img
  }
  
  features <- conv_base %>%
    predict(inputs)
  
  flatten_features <- features %>%
    array_reshape(c(num.files, 4*4*512))
  
  feature.df <- tibble(dir=68, path=paste0(directory,all.files), category=class_name, features=split(flatten_features, 1:nrow(flatten_features)))
  
  return(feature.df)
}

# Specify directory file path
dirs_68 <- list.dirs(path="Scenes/68-scenes")[2:64]

# Find classes from file names 
classes <- str_extract(dirs_68, pattern="\\w+(?=-68)") 

features <- NA

for(i in 1:length(classes)){
  class_features <- extract_features(dirs_68[i],classes[i])
  if(is.na(features)){
    features <- class_features
  } else {
    features <- rbind(features, class_features)
  }
}

save(features, file="scene_features_68_scenes.Rdata")

f <- table(features$category)

