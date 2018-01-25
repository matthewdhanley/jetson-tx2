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

