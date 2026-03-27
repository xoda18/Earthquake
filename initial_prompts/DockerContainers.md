## Earthquake
docker file to run earthquake sensor. When detects the earthquake,it sends a signal to swarms.



## Drone
Once the button "launch a drone" in UI is pressed, excute this code 
It will shoot 4 pictures and upload them to database

## DB management Somewhere here 
Remove old images, leaving only two versions. 

## Image diff 
After the images are presented in a database launch image diff. 

## VLM
Image diff produces ouput resized
Afther that a VLM is analyzing the cracks 


We need to manage all of that. I propose to have some orhestrator and the thing that 
Afther finishing all the steps each code will send an API post to some server 
And then the server will launch the next docker container to make some work done
