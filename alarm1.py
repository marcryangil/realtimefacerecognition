import pyttsx3
import threading

alarm_sound = pyttsx3.init()
voices = alarm_sound.getProperty('voices')
alarm_sound.setProperty('voice', voices[0].id)
alarm_sound.setProperty('rate', 150)

def voice_alarm(alarm_sound):
    alarm_sound.say("No Mask Detected")
    alarm_sound.runAndWait()


alarm = threading.Thread(target=voice_alarm, args=(alarm_sound,))


alarm.start()



alarm_sound.stop()

