import copy
import math
import sys

from simulation.SnetAgent import SnetAgent


class Exogenous(SnetAgent):
    # All exogenous agents, including Human, subclass off this class
    # By exogenous, we mean that the messages that are entered are scheduled beforehand
    # independently of the events of the simulation.  For now, one can put an initial
    # message time, a final message time, and a period.
    def __init__(self, unique_id, model, message, parameters):
        super().__init__(unique_id, model, message, parameters)
        self.message = message
        self.model.blackboard.append(self.message)
        self.initial_trade_plan = copy.deepcopy(self.message)

    def step(self):
        print("In Exogenous step," + self.b[self.unique_id]['label'] + " time " + str(self.model.schedule.time))
        step = math.floor(self.model.schedule.time)
        initial_message = self.initial_trade_plan['initial_message'] \
            if 'initial_message' in self.initial_trade_plan else 0
        final_message = self.initial_trade_plan['final_message'] \
            if 'final_message' in self.initial_trade_plan else sys.maxsize
        message_period = self.initial_trade_plan['message_period'] \
            if 'message_period' in self.initial_trade_plan else 1

        if initial_message <= step <= final_message and step % message_period == 0:
            new_message = copy.deepcopy(self.initial_trade_plan)
        else:
            new_message = self.blank_message()
        self.set_message(new_message)

    def payment_notification(self, agi_tokens, tradenum):
        # Pre-existing camelCase paymentNotification didn't override the
        # snake_case abstract method on SnetAgent, leaving Exogenous
        # uninstantiable. Renamed to match the base class.
        pass

    def buyer_score_notification(self, score, tradenum):
        # Static buyers don't accumulate CMA-ES fitness — they're stable
        # demand sources for sellers to learn against. No-op satisfies the
        # abstract method contract.
        pass

    def seller_score_notification(self, score, tradenum):
        pass
