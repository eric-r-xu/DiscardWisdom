### README: Autoplay Python Script

[Flow Schematic](https://github.com/eric-r-xu/DiscardWisdom/blob/main/HKMJ%20Decision%20Tree.png)

#### 1. Start the Python Script
Open a terminal window and run:
```zsh
version=11; cd /Users/ericxu/Documents/Jupyter/mahjong; conda activate myenv; ts=$(date +%s); python autoplay_v${version}.py 2>&1 | tee logs/log_$ts.txt

# to run in background, use this instead
version=11; cd /Users/ericxu/Documents/Jupyter/mahjong; conda activate myenv; ts=$(date +%s); nohup python autoplay_v${version}.py 2>&1 | tee logs/log_$ts.txt &
```

#### 2. Start the Mahjong Game
On an Android device (tested on Google Pixel 8 Pro), open the [Hong Kong Mahjong Club App](https://play.google.com/store/apps/details?id=com.pvella.mahjong&hl=en_US) and start a game.

#### 3. Mirror the Mobile Screen
In another terminal window, run:
```zsh
scrcpy --video-bit-rate 16M
```

#### 4. Enter Full Screen
Use the green circle or press `fn+F5` on a Mac keyboard to enter full screen.

#### 5. Stop the Autoplay Script
To stop all `autoplay_` Python processes (if run in background), run:
```zsh
ps aux | grep autoplay_ | grep -v grep | awk '{print $2}' | xargs kill
```
