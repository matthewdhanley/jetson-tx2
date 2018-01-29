# Getting started with the Jetson TX2
## What is the Jetson TX2?
The NVIDIA Jetson TX2 is a single board computer created by NVIDIA. It could be thought of as a supercomputer for the masses. Unfortunately, it's not as simple as plugging in a monitor and pressing the power button to reach the full potential of the TX2. 

### Some Specs
* NVIDIA Parker series Tegra X2: 256-core Pascal GPU and two 64-bit Denver CPU cores paired with four Cortex-A57 CPUs in an HMP configuration
* 8GB of 128-bit LPDDR4 RAM
* 32GB eMMC 5.1 onboard storage
* 802.11b/g/n/ac 2x2 MIMO Wi-Fi
* Bluetooth 4.1
* USB 3.0 and USB 2.0
* Gigabit Ethernet
* SD card slot
* SATA 2.0
* Complete multi-channel PMIC
* 400 pin high-speed and low speed industry standard I/O connector
* NVIDIA Pascal, 256 CUDA cores
* And MORE!!!

To someone who is just getting started with computers such as the TX2, these features may seem quite foreign. The purpose of this study is to dig into the weeds of the Jetson TX2 and explain what some of these feature are and why they can be useful.

On the TX2 exists Pascal GPU cores. These cores are also used in video cards used with virtual reality and 4k 3D gaming. They allow the Jetson to do a massive amount of computing work at once while taking up a relatively small amount of power.

## How to get it running?
NVIDIA provides a failry good [guide](http://developer2.download.nvidia.com/embedded/L4T/r27_Release_v1.0/Docs/Jetson_X2_Developer_Kit_User_Guide.pdf?i_NII0fO09Qddrnp8XafYkSLfI8kSd0CBHNbrTEXeyWpnC4bh0pRJWKO1YYJIb7pfR_9ZZGQ7bOICqt2RMjqHIJR7Mpy18x5C8ZKlwg-Gc3OJyIQDhI3-91QKH_H5lowDr8ayZ-x_8_rN1qLjtfVjCxwesTwO6VdyigIWHR_3RpCW5f_WXkJ1g) to explain how to get the TX2 up and running.

### A few things to note when starting up
* You must use a HDMI compatable monitor. HDMI to VGA or HDMI to DVI will likely not work with the TX2. The Jetson will only work with VGA if its default resolution coinsides with your monitor's resolution. This means that the Jetson needs to otherwise query the monitor for specs, but it cannot do that through VGA or DVI.

## JetPack
JetPack is available for download [here] (https://developer.nvidia.com/embedded/jetpack).

### What is JetPack?

### How Do I Get JetPack?
JetPack is required to be downloaded onto a "host" computer before it is flashed onto the TX2. Why? JetPack is designed to run a x86_64 Linux PC, but the Jetson has an [ARM architecture](https://en.wikipedia.org/wiki/ARM_architecture). Note that it is reccommended this process is performed with a host machine running Ubuntu 14.04. I performed it using Ubuntu 16.04.

1. Download [JetPack](https://developer.nvidia.com/embedded/jetpack) from NVIDIA. (I downloaded JetPack 3.1). You may need to make an account with NVIDIA to download. This downloads a file called something along the lines of `JetPack-L4T-3.1-linux-x64.run`.

2. In the directory you downloaded this file (mine is in `/home/matt/Downloads/`), change the permissions of the download to add execution permissions. I moved the executable to a new folder in my home directory `/home/matt/jetson_installer`
```
chmod +x JetPack-L4T-3.1-linux-x64.run
```

3. Run the executable from the directory within the executable is located
```
./JetPack-L4T-3.1-linux-x64.run
```
This will open the installer which should look like the image below.
![Installer](https://github.com/matthewdhanley/jetson-tx2/getting_started/img/Installer.png)

4. Follow the on-screen instrunctions. It will verify the install location and prompt you to choose your board. It will ask for an administrative password then prompt you what components you would like to download. I selected a full install and selected "Automatically resolve dependency conflicts" on the bottom left of the window before selecting next.

![Component Manger]
(https://github.com/matthewdhanley/jetson-tx2/tree/master/getting_started/img/component_manager.png)

Read and accept the "Terms and Conditions" then the downloads will begin. The downloads took about 7 minutes on a 170 mbps wifi connection. Once the downloads finish, the installs will begin.

I was prompted with an error to install OpenCV for Tegra manually manually before continuing.
![OpenCV Error]
(https://github.com/matthewdhanley/jetson-tx2/tree/master/getting_started/img/opencv_error.png)

I opened a terminal on the host and installed manually.
```
sudo apt-get install libopencv4tegra libopencv4tegra-dev
```
Once this finished, I selected "OK" on the error then "Next" on the Component Manager and the install continued nominally.

You should see this screen once the install is finished. Select "Next."

5. You will be prompted to select you "Network Layout." I selected the first choice as I have my Jetson and Host both connected to the internet via wifi. Click "Next."

I was prompted to "select the network interface on host that connects to the same router/switch as the target device."

Two options were given to me, wlp5s0 and enp4s0. To determine the correct one, I ran `ifconfig` to see my network interfaces. Sure enough, these two were present. I then ran the following two commands to determine which one was the correct one.
```
ping -c3 -I enp4s0 www.google.com
ping -c3 -I wlp5s0 www.google.com
```
The enp4s0 resulted in a "Destination Host Unreachable" message, thus this cannot be the correct network interface for the host machine. I selected wlp5s0. Note: this information is needed for ssh.
![ifconfig]
(https://github.com/matthewdhanley/jetson-tx2/tree/master/getting_started/img/if_config.png)

6. Before you start this step, make sure your Jetson TX2 wired to your router.
You will be presented with this screen before the install starts.
![Post Install]
(https://github.com/matthewdhanley/jetson-tx2/tree/master/getting_started/img/post_installation.png)
When you hit next, a shell will appear telling you how to put your device into Force USB Recovery mode. 
force usb recovery
30 gb volume popup (hangs for a bit)
system.img (takes a bit)
finished flashing (now the jetson boots up to ubuntu)
"Determining the IP address of the target..." (NEEDS TO BE CONNECTED TO ETHERNET!!!)
fin.





