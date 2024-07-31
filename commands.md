# Commands to run autoplay python script (tested on MacBook Pro (Retina, 13-inch, Early 2015) with macOS Monterey version 12.7.5)

1. in one terminal window, run

```zsh
cd /Users/ericxu/Documents/Jupyter/mahjong;conda activate myenv;ts=$(date +%s);nohup python autoplay_v7.py >> /logs/output_${ts}.log 2>&1 &
```

2. on an android mobile device (tested on google pixel 8 pro), open the [Hong Kong Mahjong Club App](https://play.google.com/store/apps/details?id=com.pvella.mahjong&hl=en_US) and start a game

3. in another terminal window, run
```zsh
scrcpy
```

4. Running #3 should open a mirrored window of your android mobile device screen.  Enter full screen (press green circle on top left to the right of the right and yellow circles) 

