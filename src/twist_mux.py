#!/usr/bin/python
import rospy
from roslib import message
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist

def main():
    rospy.init_node('TwistMux')
    tm = TwistMux()
    while not rospy.is_shutdown():
        pass

class TwistMux:
    def __init__(self):
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.joy_sub = rospy.Subscriber('/joy_vel', Twist, self.joy_vel_callback)
        self.pause_sub = rospy.Subscriber('/joy_pause', Bool, self.joy_pause_callback)
        self.paused = False

    def joy_vel_callback(self, data):
        if self.paused:
            t = Twist()
            self.pub.publish(t)
        else:
            self.pub.publish(data)

    def joy_pause_callback(self, data):
        self.paused = bool(data.data)

if __name__ == "__main__":
    main()