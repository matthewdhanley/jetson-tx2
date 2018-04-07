# Messing with Mat (container)
Mat container acts very similarily to reference counting pointer.
When you create another image of the "Mat" type, call "copy constructor," or use assignment operator, no new memory is allocated. The variables just become references to the SAME data. Note that running my function "draw_circle_square" draws a rectangle on image1 and a circle on image0, but displaying image0 shows both a square and circle! The same effect will be seen if "image2" is displayed.


