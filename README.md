# StringArt package

This is the repository of my little coding and crafting project. I created class that turns any picture into a cring art on circular frame with defined number of nails. Not all images are suitable for this - the picture that usually works best are pictures that has high contrast, not so many sharp edges. Faces and animals tend to work well. Landscapes, buiodings and similar things usually don't. 

The class itself is in a string_art.py file and notebook tutorial.ipynb demonstrates the basic functionality. File development_notebook.ipynb is my notebook that I used for development and debugging.

Here is the example of an image that I decided to turn into string art:

<img src="https://github.com/otheiner/thread_art/blob/main/assets/photo.png" width="450">

And this is how the art is made. The string starts on nail 0 (you can see the line starting on the right side of the frame) and then it continues as one continuous thread. The first 100 lines is displayed below:

<img src="https://github.com/otheiner/thread_art/blob/main/assets/string_art_236_nails_100_lines.png" width="450">

... 1500 lines ...

<img src="https://github.com/otheiner/thread_art/blob/main/assets/string_art_236_nails_1500_lines.png" width="450">

... 3000 lines ...

<img src="https://github.com/otheiner/thread_art/blob/main/assets/string_art_236_nails_3000_lines.png" width="450">

And this is the snippet of the instuctions which nails to connect for the first 100 lines. It starts at nail 0, then goes to nail 182, then 154, ... The whole sequence is in ```/assets/output_sequence_236_nails.txt```.

```
----- Lines 0 to 25 ----- 

[  0 182 154 168 156]
[84 56 78 50 80]
[58 62 59 76 52]
[85 48 80 45 82]
[54 78 58 61 57] 

 ----- Lines 25 to 50 ----- 

[ 73 135  73  45  86]
[159  87 159  87  49]
[82 42 78 51 91]
[164  94 161  90 160]
[ 91 162  89  48  84] 

 ----- Lines 50 to 75 ----- 

[160  92 162  86  53]
[ 71  57 101  59  76]
[139  75 134  74  52]
[ 89 161  92  49  77]
[142  75  44  85 160] 

 ----- Lines 75 to 100 ----- 

[ 90 163  88  45  79]
[146 168 154 170 152]
[ 80  55  73 137  77]
[ 35  73  58 104  61]
[110  62 112  64 115]
```

I can also generate images which have hole in them (this is me with open mouth), so one can get a little bit creative. The way of doing it is decribed in the tutorial notebook.

<img src="https://github.com/otheiner/thread_art/blob/main/assets/open_mouth.png" width="450">

And another image that I generated - this time not a face.

<img src="https://github.com/otheiner/thread_art/blob/main/assets/dog.png" width="450">

