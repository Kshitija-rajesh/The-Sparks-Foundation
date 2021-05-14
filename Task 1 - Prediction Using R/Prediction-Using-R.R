#Task 1-- Predict the percentage of an student based on no. of hours
#take data from csv file
r=read.csv
print(r)
relate<-lm(Scores~Hours, data=r)     ##create relationships and get the coeffiencts
print(relate)
summary(relate)  ##get the summary of the data

#prediction of  percentage score when student study for 9.25h/day---->

a<-data.frame(Hours=9.25)
perc<-predict(relate,a)
print(perc)  #Hence 92.90985%

#visulaize regression graphically

png(file="linearregession.jpg") #create file
plot(r$Hours,r$Scores,main="Study hour and score",col="green",
    abline(relate),xlab="Hours",ylab="Scores",xlim=c(2,15),ylim=c(20,100),pch=16)

dev.off()

