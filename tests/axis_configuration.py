import matplotlib.pyplot as plt

from humolire.PDR.dataloaders import load_ronin_txts
from humolire.PDR.visualize import plot_sequence

"""
This test unit identifies which axis is the one defined by an already known sequence. i.e. if you move your IMU 
with a roll motion, you would expect the Z axis to be the one changing 

You may want to download the testsuite provided.
"""


def test_accelerations():
    """
    In this test sequence, the phone is held on portrait mode, screen facing the user and usb pointing to the ground
    """
    time, acce, gyro = load_ronin_txts("test_sequences/acceleration_x")
    # y axis should present a 2 huge accelerations that represent a jump then oscillations
    # x axis should show a step pattern stronger and more accurate than Z axis
    plot_sequence(acce, time, title="acceleration_on_x")
    plt.show()

    time, acce, gyro = load_ronin_txts("test_sequences/acceleration_y")
    # y axis should decrease, increase, stay still, increase then decrease : an elevator going from floor 2 to 0 then
    # back. x and Z axis should have static noise only  and remain constant
    plot_sequence(acce, time, title="acceleration_on_y")
    plt.show()

    time, acce, gyro = load_ronin_txts("test_sequences/acceleration_z")
    # y axis should present a 2 huge accelerations that represent a jump then oscillations)
    # Z axis should show a step pattern stronger and more accurate than x axis
    plot_sequence(acce, time, title="acceleration_on_z")
    plt.show()


def test_rotations():
    """
    In this test sequence, the phone is held on portrait mode, screen facing the user and usb pointing to the ground
    """
    time, acce, gyro = load_ronin_txts("test_sequences/rotation_x")
    # phone is on table, face towards my left. turned 180° turn in the positive direction
    plot_sequence(gyro, time, title="rotation_on_x")
    plt.show()

    time, acce, gyro = load_ronin_txts("test_sequences/rotation_y")
    # phone is in my hand held against the table, front camera facing me. Span it 180° positive
    plot_sequence(gyro, title="rotation_on_y")
    plt.show()

    time, acce, gyro = load_ronin_txts("test_sequences/rotation_z")
    # phone is on table, front camera face to the ceiling  and span 360° positive
    plot_sequence(gyro, title="rotation_on_z")
    plt.show()


if __name__ == '__main__':
    test_accelerations()
    test_rotations()
    plt.show()
    plt.close()
