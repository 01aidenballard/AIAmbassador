import sys
import os
import qrcode

from PIL import Image

# sys.path.append("/home/ubuntu/Robotics/QuadrupedRobot")
# sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("/home/ubuntu/Robotics/QuadrupedRobot") for name in dirs])
# from Mangdang.LCD.ST7789 import ST7789

class Display:

    def generate_redirect_qr_code(answer_source):
        """
        Generate a QR code that redirects to the given answer source URL.
        Args:
            answer_source (str): The URL to encode in the QR code.
        Returns:
            None
        """

        # Take the source link from the answer source
        source = answer_source

        # Remove the .txt suffix if it exists
        true_source = source.removesuffix(".txt")

        # Generate the QR code
        img = qrcode.make(true_source)

        # Save the QR code image to our interface folder
        img.save("../Interface/Images/direct_to.png")

        return
    
    def display_qr_code():
        """
        Display the generated QR code on the ST7789 display.
        Args:
            None
        Returns:
            None
        """

        # init st7789 device 
        disp = ST7789()
        disp.begin()
        disp.clear()

        # show exaple picture
        image=Image.open("../Interface/Images/direct_to.png")
        image.resize((320,240))
        disp.display(image)
    
        return
    