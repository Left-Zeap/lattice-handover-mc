from __future__ import annotations
import math

def trapezoid_parameters(distance: float, duration: float, acceleration: float):
    """
    Symmetric acceleration / cruise / deceleration profile.

    distance = a * t_acc * (duration - t_acc)
    """
    a = abs(float(acceleration))
    if a <= 0:
        raise ValueError("acceleration must be > 0")
    disc = duration * duration - 4.0 * distance / a
    if disc < 0:
        raise ValueError(
            "requested distance/duration cannot be reached with given acceleration"
        )
    t_acc = 0.5 * (duration - math.sqrt(disc))
    t_cruise = duration - 2.0 * t_acc
    vmax = a * t_acc
    return t_acc, t_cruise, vmax

def trapezoid_kinematics(t, distance, duration, acceleration, start=0.0):
    """
    Return position q, velocity v and acceleration a at scalar time t.
    """
    amag = abs(float(acceleration))
    t_acc, t_cruise, vmax = trapezoid_parameters(distance, duration, amag)
    t = max(0.0, min(float(t), duration))

    if t < t_acc:
        acc = amag
        v = amag * t
        x = 0.5 * amag * t * t
    elif t < t_acc + t_cruise:
        acc = 0.0
        v = vmax
        x = 0.5 * amag * t_acc**2 + vmax * (t - t_acc)
    else:
        td = t - (t_acc + t_cruise)
        acc = -amag
        v = vmax - amag * td
        x0 = 0.5 * amag * t_acc**2 + vmax * t_cruise
        x = x0 + vmax * td - 0.5 * amag * td**2

    return start + x, v, acc

def linear_ramp(t, duration, start, stop):
    if duration <= 0:
        return stop
    f = min(1.0, max(0.0, float(t) / duration))
    return start + (stop - start) * f
