
I need a drone to scan the wall. We do four blocks with 5 april tags in total 
such that when drone takes a photo each image has two april tags in the opposite corners)
So it is like that
A--------------A---------------|
|       |      ||       |      |
|  IMG1 | IMG2 ||  IMG3 | IMG4 |
|       |      ||       |      |
--------A------|--------A------A  

where A is just some april tags (ID is not defined). 

I need to fly the drone to scan the wall. 
Is should rise go for two metres heights. 
Then go for 1.5 metre to the right, take a picture and save it (now in folder with date and time name, but in future to databse) 
And i want it to repeat 4 times in total to take four images, then the drone should return back and land. 
Sleep between the actions.

The problem is with taking images. You need to look at rpsm protocol and extract the Key frame from it.
The problem is that we are using android sdk and we have a 20 sec latency. We use mediamtx which transforms rtsmp portocol to all the others
On a phone there is no latency, but on pc  there is. 
We do not know how to fix it.

The other approach could be as i soo just use function captureShot but it wiil probably send this to the drone memory and 
we will not be able to access it from pc. 

So what do u think with the image. How to solve this problem ,what to do, should i give you more context???
Once we solve the problem you will write a code to contrlo the drone and save the image.


Look at DJIContaloClient.py too see what code we have. 
To get video stream i use http://localhost:8889/live/webcam/