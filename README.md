Self driving car lane and path detection
=========================================
## Running OpenPilot on GTAV 
This project is a fork from [littlemountainman/modeld](https://github.com/littlemountainman/modeld) 

## How to install

To be able to run this, I recommend using Python 3.7 or up.

1. Install the requirements 

```
pip3 install -r requirements.txt
```

This will install all the necessary dependencies for running this. 

2. Download [Vpilot with DeepGTAV](https://github.com/aitorzip/VPilot)

3. Download [ScriptHook](https://www.dev-c.com/gtav/scripthookv/)

4. Download [DeepGTA](https://github.com/aitorzip/DeepGTAV)

5. Copy ScriptHookV.dll, dinput8.dll, NativeTrainer.asi to the game's main folder, i.e. where GTA5.exe is located.

6. Copy-paste the contents of DeepGTAV/bin/Release under your GTAV installation directory

7. Launch the program and GTAV is already running
```
python3 main.py
```


## Credits

[littlemountainman/modeld](https://github.com/littlemountainman/modeld)
[aitorzip/DeepGTAV](https://github.com/aitorzip/DeepGTAV)
[aitorzip/VPilot](https://github.com/aitorzip/VPilot)

