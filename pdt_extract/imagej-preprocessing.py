import imagej
import scyjava
scyjava.config.add_option('-Xmx6g')

import os
#os.environ['JAVA_HOME']='/usr/local'

ij = imagej.init("/content/Fiji.app/")
# Open an image
imp = ij.io().open("../Pendant Drops/d.0.55.png")