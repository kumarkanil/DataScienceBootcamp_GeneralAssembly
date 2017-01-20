#!/usr/bin/env python

'''
GA Data Science Q2 2016

Code walk-through 17: Discrete event simulation
'''

import random

import numpy as np
import simpy

import matplotlib.pyplot as plt

%matplotlib qt

class Customer(object):
    def __init__(self, env, washing_machines, dryers, baskets):
        self.env = env
        env.process(self.do_laundry(washing_machines, dryers, baskets))

    def do_laundry(self, washing_machines, dryers, baskets):
        # Save arrival time
        self.arrival_time = self.env.now

        # Request a washing machine
        wm_req = washing_machines.request()
        yield wm_req

        # Save start time
        self.start_time = self.env.now

        # Wait 20 minutes (constant) for washing machine to finish
        yield self.env.timeout(20)

        # Request a basket
        b_req = baskets.request()
        yield b_req

        # Take between 1 and 3 minutes to unload the washing machine, then
        # release it (but keep the basket)
        yield self.env.timeout(random.uniform(1, 3))
        washing_machines.release(wm_req)

        # Request a dryer
        d_req = dryers.request()
        yield d_req

        # Take between 1 and 3 minutes to load the dryer, then release the
        # basket (but keep the dryer)
        self.env.timeout(random.uniform(1, 3))
        baskets.release(b_req)

        # Wait between 10 and 15 minutes for dryer to finish, then release it
        yield self.env.timeout(random.uniform(10, 15))
        dryers.release(d_req)

        # Save end time
        self.end_time = self.env.now

    @property
    def waiting_time(self):
        return self.start_time - self.arrival_time

    @property
    def laundry_time(self):
        return self.end_time - self.start_time

class Launderette(object):
    def __init__(self, n_washing_machines, n_dryers, n_baskets, n_customers):
        self.env = simpy.Environment()
        self.washing_machines = simpy.Resource(self.env,\
                                               capacity=n_washing_machines)
        self.dryers = simpy.Resource(self.env, capacity=n_dryers)
        self.baskets = simpy.Resource(self.env, capacity=n_baskets)
        self.n_customers = n_customers
        self.env.process(self.arrivals())
        self.env.run()

    def arrivals(self):
        self.customers = []
        while len(self.customers) < self.n_customers:
            # Wait for next customer to arrive
            yield self.env.timeout(random.expovariate(0.2))
            self.customers.append(Customer(self.env, self.washing_machines,\
                                           self.dryers, self.baskets))

    @property
    def arrival_times(self):
        return np.array([ c.arrival_time for c in self.customers ])

    @property
    def waiting_times(self):
        return np.array([ c.waiting_time for c in self.customers ])

    @property
    def laundry_times(self):
        return np.array([ c.laundry_time for c in self.customers ])

# On average, a customer arrives every 5 minutes.
# Whenever a customer arrives, they request a washing machine and queue if none
# are available.
# The washing cycle takes 20 minutes (constant).
# At the end of the washing cycle, the customer requests a basket (queuing if
# none are available) and unloads the washing machine; this takes between 1 and
# 3 minutes (uniformly distributed).
# With laundry in the basket, the customer requests a dryer (queueing if none
# are available) and loads it; this also takes between 1 and 3 minutes
# (uniformly distributed).
# The drying cycle takes between 10 and 15 minutes (uniformly distributed).
#
# Questions:
# 1) How long do customers have to queue for a washing machine when they arrive?
# 2) How long does it take them to do laundry?

# Simulate a launderette with 5 washing machines, 4 dryers, and 3 baskets
l1 = Launderette(5, 4, 3, 1000)

# Extract times
wt = l.waiting_times()
lt = l.laundry_times()

# Plot times by customer
plt.plot(l.waiting_times())
plt.plot(l.laundry_times())

# Mean waiting time
np.mean(wt)

# Percentage of customers who had to wait
np.mean(wt > 0)

# Mean waiting time among customers who had to wait
np.mean(wt[wt > 0])

# Mean laundry time
np.mean(lt)

