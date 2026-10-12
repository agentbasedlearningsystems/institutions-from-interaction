import os
import bisect
import time
import random
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
from mesa import Agent
from sklearn.metrics.pairwise import cosine_similarity
import copy
import sys
import re


def stop_cut(name):
    """Cut a name at its stop codon.

    HER RULE (2026-08-14): the codon is the whole underscore-delimited
    segment 'stop'. Cut at the FIRST one - at the end of the name
    ('vectorSpace_stop') or in the middle ('x_stop_y'), everything from
    the codon on is unread. A longer word that merely begins with the
    letters s-t-o-p ('stopwords') is not the codon and is kept: cutting
    inside it is what made preprocessor_freetext_stopwords a name no
    program could ever carry.

    A bare 'stop' stays 'stop': it is the plan-ending sentinel, and the
    callers that key on it test for it by name.
    """
    parts = str(name).split('_')
    if 'stop' in parts:
        idx = parts.index('stop')
        if idx > 0:
            return '_'.join(parts[:idx])
        return 'stop'
    return str(name)


class MarketAgent(Agent, ABC):
    def __init__(self, unique_id, model, message, parameters):
        # In the subclass, make sure to submit an initial message to the blackboard, if the message field is blank.
        # When the simulation initializes, the first thing created are the agents with initial messages defined in
        # the configuration file.  In these cases, the message field to this routine is filled in with the agent's
        # message from the config file and the unique id of the agent is the order in the file. Next, agents are
        # created that do not have a specified message in the configuration file. The parameters of the config file
        # refers to them as the random_agents, and specifies the number to generate of each. A function is supplied
        # ,float_vec_to_trade_plan, that converts a vector of floats to a message, that could be a random vector.
        # However, it is up the individual agent what initial message to submit to the board. Regardless of the origin
        #  of their initial message,  agents are parameterized in the config file parameters under the name
        # agent_parameters.  These parameters are tailored to their subclasses and come into this routine as the
        # "parameters" parameter.

        # Asynchronous Optimization (an ask and tell interface) is needed in subclasses because MarketSim
        # has the program control

        super().__init__(model)
        self.unique_id = unique_id  # override Mesa 3.x auto-assigned id
        self.message = message
        self.p = self.model.parameters  # convenience method:  its shorter
        self.b = self.model.blackboard  # convenience method:  its shorter
        self.o = self.model.ontology  # convenience method:  its shorter
        self.parameters = parameters
        self.wealth = 0
        self.emergent_pattern = re.compile(r'^f\d+\.\d+_\d+_\d+_\d+')
        self.item_type_pattern = re.compile(r'^([a-zA-Z0-9]+)_?')
        self.test_item_type_pattern = re.compile(r'^test_([a-zA-Z0-9]+)_?')




    @abstractmethod
    def step(self):
        # Each agent has a message slot on the blackboard, and the purpose of this step is to submit a new message
        # into this slot, if the subclassed agent so desires.  Human agents do not submit new messages during the run
        #  of the simulation, which operates on a different time scale than they do.  Machine learning agents that
        # submit a message here can see the detailed results of their message added to that message the next time
        # this function is called.  These results include the entire blackboard, with each agent indicating the
        # messages they bought from self, selfs test scores, and the money given to self.  All this is taken account
        # in the net AGI tokens, which is available to agents to use as part of the reward for reinforcement learning
        #  or machine learning. The change in tokens from the last time step has been called is the result of the
        # previous message.  A notification is sent that this agent can use to keep track of the net,
        # in the payment_notification convenience method. For the final step the agent can observe the results of the
        # last and submit None to the blackboard.
        pass

    @abstractmethod
    def payment_notification(self, agi_tokens, tradenum):
        # This routine is called to notify the agent that his wealth has been changed by an agi_tokens amount,
        # which can be negative. The blackboard may be examined for more information on which parts of the trade
        # plans resulted in what payments, test scores, etc.  This is called after each
        pass

    @abstractmethod
    def seller_score_notification(self, score, tradenum):
        # This routine is called to notify the agent that his wealth has been changed by an agi_tokens amount,
        # which can be negative. The blackboard may be examined for more information on which parts of the trade
        # plans resulted in what payments, test scores, etc.  This is called after each
        pass

    @abstractmethod
    def buyer_score_notification(self, score, tradenum):
        # This routine is called to notify the agent that his wealth has been changed by an agi_tokens amount,
        # which can be negative. The blackboard may be examined for more information on which parts of the trade
        # plans resulted in what payments, test scores, etc.  This is called after each
        pass


    def price_overlap(self, buy, sell):
        # the curator's double-auction law (restated 2026-07-20): each side
        # has a learned band, midpoint +/- range. No band intersection,
        # no sale. A missing range means a band of width zero.
        buy_low = max(buy['midpoint'] - buy.get('range', 0.0), 0.0)
        buy_high = buy['midpoint'] + buy.get('range', 0.0)
        sell_low = max(sell['midpoint'] - sell.get('range', 0.0), 0.0)
        sell_high = sell['midpoint'] + sell.get('range', 0.0)
        return buy_high >= sell_low and sell_high >= buy_low

    def band_distance(self, buy, sell):
        """How far apart the two price bands are: zero when they overlap
        (a trade could clear), otherwise the gap between them.

        Her ruling (2026-08-12, a18): price stops being a wall and
        becomes a pull. This is the distance the draw weight falls off
        with. The bands are read exactly as price_overlap reads them, so
        a distance of zero and an overlap mean the same thing.
        """
        buy_low = max(buy['midpoint'] - buy.get('range', 0.0), 0.0)
        buy_high = buy['midpoint'] + buy.get('range', 0.0)
        sell_low = max(sell['midpoint'] - sell.get('range', 0.0), 0.0)
        sell_high = sell['midpoint'] + sell.get('range', 0.0)
        if buy_high >= sell_low and sell_high >= buy_low:
            return 0.0
        return float(max(sell_low - buy_high, buy_low - sell_high))

    def price(self, buy, sell):
        # The agreed price is the midpoint of the bands' intersection
        # (her law: "if there is no overlap, no sale, but if there is,
        # take the midpoint").
        buy_low = max(buy['midpoint'] - buy.get('range', 0.0), 0.0)
        buy_high = buy['midpoint'] + buy.get('range', 0.0)
        sell_low = max(sell['midpoint'] - sell.get('range', 0.0), 0.0)
        sell_high = sell['midpoint'] + sell.get('range', 0.0)
        lo = max(buy_low, sell_low)
        hi = min(buy_high, sell_high)
        if hi < lo:
            lo = hi = (buy['midpoint'] + sell['midpoint']) / 2.0
        price = (lo + hi) / 2.0

        return price

    def _would_pay(self, buy, best):
        """HER FIXED MONEY (4 Oct 2026): what this settle would cost the buyer, computed exactly as the
        settle computes it with learned prices (the band law's price, her zero-pay ruling, the quality
        law), before any record is written. The settle checks its own price against this and stops the
        run on any difference."""
        max_score, winning_num = best[0], best[1]
        _off = buy['offers'][winning_num]
        sell = self.b[_off['agent']]['trades'][_off['trades'][0]]
        price = self.price(buy, sell)
        if self.p.get('chooser_runnable_closest') or self.p.get('chooser_roulette'):
            _zp_live = False
            for _td0 in buy.get('tests', []):
                if _td0.get('stophere'):
                    break
                if stop_cut(str(_td0.get('test', ''))) and stop_cut(str(_td0.get('data', ''))):
                    _zp_live = True
                break
            if _zp_live and max_score <= 0 and self.p.get('pay_by_score', True):
                return 0.0
        if max_score > 0 and self.p.get('pay_by_score', True):
            price *= self.quality_factor(max_score, max_score)
        return float(price)

    def _money_row(self, buy, amount, outcome):
        """One row per purchase attempt of a human with fixed money (CORE_MONEY_DEBUG): what it would
        cost, what was left after, and whether it was paid or stopped for lack of money."""
        _mp = os.environ.get('CORE_MONEY_DEBUG')
        if not _mp:
            return
        try:
            _new = not os.path.exists(_mp)
            with open(_mp, 'a') as _f:
                if _new:
                    _f.write('time;buyer;good;amount;money_left;outcome\n')
                _f.write('%s;%s;%s;%.6f;%.6f;%s\n' % (getattr(self.model.schedule, 'time', -1), self.unique_id,
                                                      buy.get('cd_good', ''), amount, self._money_left, outcome))
        except OSError:
            pass

    def set_message(self, message):
        self.message = message
        self.model.blackboard[self.unique_id]=self.message

    def quality_factor(self, score, frontier):
        """Payment factor = score^k, k = quality_exponent (her ruling
        2026-10-02, "the former": pay by quality against the ideal 1.0, no
        frontier in payment - the frontier's history (Jul 14 pioneer
        premium) was read and weighed: ghosts of dead pioneers, a bias
        within generations, yardsticks frozen after 100 boards). The
        frontier record is kept as a MEASURE only. Scores are [0,1] in
        this registry; zero pays zero."""
        k = float(self.p.get('quality_exponent', 1.0))
        if score <= 0.0:
            return 0.0
        return max(0.0, min(1.0, float(score))) ** k

    def gather_offers(self):
        # for every buy offer an agent has, look for sell offers from other agents for an item that is the same
        # category asked for, for which there is an overlap in price. One can tell the same category because the
        # ontology name begins in the same way list the possible trades to be considered in the individual buy trades
        #  of the agents tradeplan. which there is overlap in price. list offers in the message, uniqueId:tradeNum
        # The lowest cosine similarity never wins, because a random cosine similarity can still be around 60,
        # and we want the ones that have learned signs to have even greater chance of succeeding.
        print("In gather_offers," + self.b[self.unique_id]['label'])

        buyer_stop_codon_reached = False
        for buy in self.message['trades']:
            if (not buyer_stop_codon_reached) and buy['type'] == 'buy' \
                    and self.p.get('sign_first_matching'):
                # HER PIPELINE (2026-08-11): exposure FIRST, gates as
                # removals - so signs must LEARN to carry category and
                # price band (the gated order never applied that
                # pressure; "this is why mi hasnt worked").
                # 1) top-n agents by cosine(buyer's sought, seller's
                #    DISPLAYED) over ALL other agents, no pre-gating;
                # 2) remove category mismatches WITHOUT replacement;
                # 3) remove price non-overlaps WITHOUT replacement;
                # survivors are the offers.
                _sought = np.array(buy['sign']).reshape(-1, len(buy['sign']))
                _cands = []
                for i, message in enumerate(self.b):
                    if i == self.unique_id:
                        continue
                    _disp = message.get('sign')
                    if not _disp:
                        continue
                    # HER RULING (2026-08-12): the draw pool is ONLY
                    # agents with a live sell listing on the board -
                    # public information, no recognition involved.
                    # Buy-only humans and non-selling learners never
                    # consume draw slots.
                    _has_sell = False
                    _sstop0 = False
                    for _tr0 in message.get('trades', []):
                        if _tr0.get('type') == 'stop':
                            _sstop0 = True
                        if _sstop0:
                            break
                        if _tr0.get('type') == 'sell':
                            _has_sell = True
                            break
                    if not _has_sell:
                        continue
                    _sv = cosine_similarity(
                        _sought, np.array(_disp).reshape(-1, len(_disp)))
                    _cands.append((float(_sv.flatten()[0]), i))
                # HER CORRECTION (same day): candidates are DRAWN at
                # random, weighted by cosine - not top-n - so
                # exploration survives and challengers get auditions.
                # ONE size, one config parameter (her ruling):
                # funnel_draws = how many are drawn before the
                # removals. ALL survivors are audited - no second
                # cap. Sized against the measured single-draw
                # survival rate.
                _n = self.p.get('funnel_draws')
                if not _n:
                    raise ValueError(
                        'sign_first_matching needs funnel_draws '
                        'in the config parameters')
                _n = int(_n)
                _pool = [(max(0.0, _s8), _i8) for _s8, _i8 in _cands]
                _drawn = []
                for _ in range(min(_n, len(_pool))):
                    _tot = sum(_w for _w, _ in _pool)
                    if _tot <= 0:
                        _idx = random.randrange(len(_pool))
                    else:
                        _rr = random.uniform(0.0, _tot)
                        _idx = len(_pool) - 1
                        for _j8, (_w8, _) in enumerate(_pool):
                            _rr -= _w8
                            if _rr <= 0:
                                _idx = _j8
                                break
                    _drawn.append(_pool.pop(_idx))
                _stop_cut = [stop_cut(buy['item'])]
                _offers = []
                for _sim, i in _drawn:
                    _sell_j = None
                    _sstop = False
                    for j, sell in enumerate(self.b[i]['trades']):
                        if sell['type'] == 'stop':
                            _sstop = True
                        if _sstop or sell['type'] != 'sell':
                            continue
                        if not sell['item'].startswith(_stop_cut[0]):
                            continue          # category removal, slot kept only if another sell matches
                        if not (self.p.get('price_list') or self.price_overlap(buy, sell)):
                            continue          # price removal (off under THE PRICE LIST, her 1 Oct ruling)
                        _sell_j = j
                        break
                    if _sell_j is None:
                        continue              # slot lost without replacement
                    _offers.append(OrderedDict([
                        ('agent', i), ('cosine_sim', _sim),
                        ('trades', [_sell_j])]))
                buy['offers'] = _offers
                continue
            if (not buyer_stop_codon_reached)and buy['type'] == 'buy':
                offers = []
                for i, message in enumerate(self.b):
                    # LOCALITY (3 Oct 2026, her Aug 8 organ, flag-gated): with
                    # locality_radius set, a buyer sees only the sellers within that
                    # ring distance on the ring of agent ids (humans and learners
                    # alike), so pointing, hiring and signs are local and distant
                    # regions may form different conventions for the same role.
                    _lr = self.p.get('locality_radius')
                    if _lr:
                        _n_ring = len(self.b)
                        _d_ring = abs(int(i) - int(self.unique_id))
                        _d_ring = min(_d_ring, _n_ring - _d_ring)
                        if _d_ring > int(_lr):
                            continue
                    if i != self.unique_id:
                        offer = None
                        seller_stop_codon_reached = False
                        for j, sell in enumerate(message['trades']):
                            if (not seller_stop_codon_reached) and sell['type'] == 'sell':
                                stop_cut_off = [stop_cut(buy['item'])]
                                # Her ruling (2026-08-08): under
                                # sign_only_matching, NO prefilter — not
                                # by role stem, not by price band. The
                                # mechanical sorting was doing the
                                # sign's job for free, which is exactly
                                # why signs never came to mean the
                                # family or carry the price. With it
                                # off, the sign similarity draw is the
                                # ONLY sorting in the market, wrong
                                # draws cost real auditions, and
                                # meaning finally has a wage.
                                # HER RULING (2026-08-12, a18): "the sign
                                # draw happens among candidates that
                                # already fit - with price no longer a
                                # gate but a weight." Category is a HARD
                                # incompatibility: a clusterer is not an
                                # embedding and no negotiation changes
                                # that, so it still culls. Price is SOFT:
                                # in a real market you do not refuse to
                                # look at someone for asking too much,
                                # you look and then pick what you can
                                # afford. So under price_weighted a
                                # seller whose band is far away is drawn
                                # LESS OFTEN rather than not at all, and
                                # the double-auction law still binds at
                                # settlement - nothing clears outside
                                # both bands. Default false keeps the
                                # wave-16 behaviour exactly.
                                # CATEGORY GATE (her ruling 2026-09-30, "filter by
                                # category so the sign always means something within
                                # the category"; the switch sign_only_matching of
                                # 16 Aug meant NO gate and is renamed off). Under the
                                # price list there is no price gate (her 1c).
                                _cat_ok = sell['item'].startswith(stop_cut_off[0])
                                _gas = self.p.get('gates_after_sign')
                                if _gas:
                                    # HER RULING (5 Oct 2026, point 1): "get rid of the filter for type
                                    # and price that happens before sign choice, so that signs can include
                                    # the information of type and price". Every sell row is an offer; the
                                    # type and price checks are recorded on the offer and applied to the
                                    # ONE drawn offer in choose_partners: a failed check is no trade this
                                    # board (her 1 Oct rule: no walk, no second draw). The type check
                                    # matches whole name segments (a buyer naming ..._creg1 must not take
                                    # ..._creg10, which the plain prefix test of the gate above allows),
                                    # and a bare stop in a buyer's row means anything.
                                    _cut_g = stop_cut_off[0]
                                    _cat_ok = (_cut_g == 'stop' or sell['item'] == _cut_g
                                               or sell['item'].startswith(_cut_g + '_'))
                                    _price_ok_g = bool(self.p.get('price_list')) or self.price_overlap(buy, sell)
                                    _pass_gate = True
                                elif self.p.get('category_gate'):
                                    _pass_gate = _cat_ok and (
                                        bool(self.p.get('price_list'))
                                        or self.price_overlap(buy, sell))
                                else:
                                    _pass_gate = self.p.get('sign_only_matching') or (
                                        _cat_ok and (self.p.get('price_weighted')
                                                     or self.price_overlap(buy, sell)))
                                if _pass_gate:
                                    # ONE OFFER PER SELL ROW (bug fix 2026-09-11,
                                    # found by her question "how do you enable more
                                    # than one sought sign for a seller, with more
                                    # than one sale?"): the old code made one offer
                                    # per SELLER, took its similarity from the first
                                    # qualifying sell row, and appended the others
                                    # to a list nobody read - every consumer used
                                    # offer['trades'][0]. Under sign_only_matching
                                    # every row qualifies, so an agent could only
                                    # ever sell its FIRST product. Now each sell row
                                    # is its own offer with its own sought sign.
                                    sought_sign = np.array(buy['sign']).reshape(-1, len(buy['sign']))
                                    displayed_sign = np.array(self.b[i]['sign']).reshape(-1, len(self.b[i]['sign']))
                                    buyers_sim = cosine_similarity(sought_sign, displayed_sign)
                                    # 7 Oct 2026: a cosine of exactly 0.0 (a zero display vector, which a language-model
                                    # learner can post) left this a 1x1 array under `if buyers_sim:` and the draw's float()
                                    # raised under NumPy 2 ("only 0-dimensional arrays"); take the scalar unconditionally
                                    buyers_sim = float(np.asarray(buyers_sim).flatten()[0]) if np.size(buyers_sim) else 0.0
                                    # the seller's OWN sought sign (this row's) against the buyer's display
                                    sought_sign = np.array(sell['sign']).reshape(-1, len(sell['sign']))
                                    displayed_sign = np.array(self.message['sign']).reshape(-1,
                                                                                         len(self.message['sign']))
                                    sellers_sim = cosine_similarity(sought_sign, displayed_sign)
                                    sellers_sim = float(np.asarray(sellers_sim).flatten()[0]) if np.size(sellers_sim) else 0.0
                                    sim = (self.p['buyers_weight'] * buyers_sim) + ((1-self.p['buyers_weight']) * sellers_sim)
                                    offer = OrderedDict([('agent', i), ('cosine_sim', sim), ('trades', [j])])
                                    if _gas:
                                        offer['_cat_ok'] = bool(_cat_ok)
                                        offer['_price_ok'] = bool(_price_ok_g)
                                    offers.append(offer)
                            elif sell['type'] == 'stop':
                                seller_stop_codon_reached = True
                buy['offers'] = offers
            elif buy['type'] == 'stop':
                buyer_stop_codon_reached = True

        # convert cosine distances into probabilities
        buyer_stop_codon_reached = False
        for buy in self.message['trades']:

            if (not buyer_stop_codon_reached) and buy['type'] == 'buy':
                if len(buy['offers']) > 1:
                    minimum = 1.0
                    for offer in buy['offers']:
                        if offer['cosine_sim'] < minimum:
                            minimum = offer['cosine_sim']
                    # The floor is subtracted RELATIVE TO THIS BUYER'S OWN
                    # candidate set, which is what makes the sign
                    # discriminating at all: two random signs in the
                    # positive unit cube are already 0.75 alike, so raw
                    # cosine gives a perfect match only a 1.37-to-1 pull.
                    # Against the set minimum it is far sharper. This was
                    # always here in the wave-16 path; the sign-first
                    # funnel weighted by raw cosine and lost it.
                    _pw = bool(self.p.get('price_weighted'))
                    _fall = float(self.p.get('price_falloff', 20.0)) or 1.0
                    # THE SIGNS EXPERIMENT (her go, 2026-08-15): two
                    # flag-gated intervention switches, one lever each.
                    # chooser_blind: offers weighted equally - the sign
                    # channel closed at the chooser while prices, types,
                    # and tests stay intact (the prices-only arm).
                    # sign_floor_subtract: weights are cosine above a
                    # FIXED floor (0.75, the geometric floor of the
                    # positive unit cube) - the strengthened-signs arm.
                    _blind = bool(self.p.get('chooser_blind'))
                    _floor = self.p.get('sign_floor_subtract')
                    weights = []
                    for offer in buy['offers']:
                        if _blind:
                            _w = 1.0
                        elif _floor is not None:
                            _w = max(offer['cosine_sim'] - float(_floor),
                                     0.0)
                        else:
                            _w = max(offer['cosine_sim'] - minimum, 0.0) \
                                ** float(self.p.get('ticket_power', 1) or 1)
                        if _pw:
                            # Price as a pull, not a wall: distance
                            # between the two bands, zero when they
                            # overlap. Far sellers stay reachable and
                            # rare; settlement still refuses a trade
                            # outside both bands.
                            _sell = self.b[offer['agent']]['trades'][offer['trades'][0]]
                            _w *= 1.0 / (1.0 + self.band_distance(buy, _sell) / _fall)
                        weights.append(_w)
                    simsum = sum(weights)
                    if simsum > 0:
                        for offer, _w in zip(buy['offers'], weights):
                            offer['probability'] = _w / simsum
                    else:
                        # Every candidate equally similar: a uniform draw,
                        # not a division by zero.
                        _u = 1.0 / len(buy['offers'])
                        for offer in buy['offers']:
                            offer['probability'] = _u
                elif len(buy['offers']) == 1:
                    buy['offers'][0]['probability'] = 1.0
            elif buy['type'] == 'stop':
                buyer_stop_codon_reached = True

                    # print(str(self.unique_id) + " in gather offers" )

    def default_leaf(self, item):
        """HER RULING (5 Oct 2026, point 2): the designated default below a name that stops early, read from the
        ontology's '_default' marks, level by level down to a leaf ("similar to parameters in the ml part"). A
        level without a designated default is an error, not a silent first child."""
        cat = stop_cut(str(item))
        for _ in range(64):
            kids = self.weights_for_level(cat)
            if not any(kids.values()):
                return cat
            d = self.retrieve_ontology_item(cat).get('_default')
            if d is None or d not in kids:
                raise ValueError('seller_name_defaults: no designated default under %r' % cat)
            cat = cat + '_' + d
        return cat

    def retrieve_ontology_item(self, cumulative_category):
        # return the ontology item from the underscore notation moregeneral_lessgeneralandmorespecific_evenmorespecific
        # it must start with the very first category
        # print('cumulative_category')
        # print(cumulative_category)
        adict = OrderedDict()
        if cumulative_category:
            levels = cumulative_category.split('_')
            adict = self.o
            #print('levels')
            #print(levels)
            for level in levels:
                if level in adict:
                    adict = adict[level]
                elif level == 'ontology':
                    pass
                else:
                    return OrderedDict()
        return adict

    def descendants(self, cumulative_category):
        # see if this ontology item is the most specific possible, which it must be for example, if it is a test and
        # is to be called with one item : the tested program, or if it is to be called and yield data.
        descendants = []
        cat_dict = self.retrieve_ontology_item(cumulative_category)

        no_descendants = True
        for name, adict in cat_dict.items():
            if not name.startswith('_') and isinstance(adict, dict):
                no_descendants = False
                if cumulative_category == 'ontology':
                    descendants.extend(self.descendants(name))
                else:
                    descendants.extend(self.descendants(cumulative_category + '_' + name))
        if no_descendants:
            descendants.append(cumulative_category)
        return descendants

    def perform_test(self, function_list):
        score = 0
        pickle_name = ""

        # THE OTHER HALF OF HER WARNING (2026-08-12). An expression that
        # still needs inputs is a PART, not a program. It may be sold, and
        # spliced into a buyer's expression where the buyer's own
        # following plans supply what it lacks - that is what "if there is
        # a buy instead of a construct, it should just continue the
        # expression with the next agent's code" means, and it is carried
        # by the deficit that function_list_arity returns.
        #
        # But a part must never be RUN as though it were finished. That is
        # what produced 9,548 starved calls in eight boards: functions
        # invoked with fewer arguments than they take, every one of them
        # logged as a broken chain. The guard belongs HERE, at evaluation,
        # not at assembly - rejecting a deficit at assembly forbids
        # selling parts at all, which I did briefly and which her warning
        # caught.
        _deficit = self.function_list_arity(function_list) if function_list else None
        # NOTE THE EXPLICIT None TEST. I first wrote `if _deficit:`,
        # meaning "unresolvable or unfinished" - but None is FALSY, so
        # unresolvable lists sailed through into gep, which does
        # `length_next_level += arity` with arity None and took the whole
        # world down. Both runs died on it.
        if _deficit is None or _deficit > 0:
            return (OrderedDict(), 0, "")

        # A BROKEN CHAIN SCORES ZERO. IT DOES NOT KILL THE WORLD.
        # That is the framework's own convention everywhere else, and it
        # was not held here: two runs died today because one unresolvable
        # chain raised inside this call. A chain rooted at something that
        # returns an array rather than a score would do the same, through
        # `if score_tuple[0]` - "the truth value of an array is
        # ambiguous". Whatever goes wrong below, the audition fails and
        # the market carries on.
        try:
            gepResult = self.modular_gep(function_list)  # put the ordered Dictionary in the global so a decorated function can access
            if any(gepResult.values()):
                root = next(iter(gepResult.items()))[0]
                score_tuple = self.model.call_emergent_function(gepResult,root)
                # MY BUG, 2026-08-12, found 2026-08-13. I "hardened" these
                # two lines with isinstance checks while adding the
                # exception containment. But pickle_name is NOT a string -
                # it is the func_tuple (root name, children), used as the
                # KEY into model.pickles, which holds the filename. A tuple
                # is never a str, so the name could never be set, so
                # buy['pickle'] was always empty, so get_bought_items
                # recorded nothing, so every run on this code delivered
                # ZERO programs: 7,548 and 27,811 report rows with not one
                # delivery, and 264 of 264 completed buys with an empty
                # name. Wave 16 tested truthiness here and worked.
                #
                # The isinstance guard did have a reason - a chain rooted at
                # something returning an ARRAY makes plain truthiness raise
                # "the truth value of an array is ambiguous" - so this keeps
                # that protection instead of reverting: the name is taken
                # when it exists, and the score is taken when it converts to
                # a number, which also recovers numpy scalars that the old
                # isinstance test silently dropped.
                if score_tuple and len(score_tuple) and score_tuple[0] is not None:
                    pickle_name = score_tuple[0]
                if score_tuple and len(score_tuple) > 1:
                    try:
                        _s = float(score_tuple[1])
                    except (TypeError, ValueError):
                        _s = None
                    if _s is not None:
                        score = _s
        except Exception as _e:
            import traceback as _tb
            try:
                if os.environ.get('MARKETSIM_WARM_CHILD'):
                    raise OSError('warm child writes no records')
                with open(self.p['output_path'] + 'caught_exceptions.log', 'a') as _h:
                    _h.write('=== perform_test refused a chain: %s: %s\nchain=%s\n%s\n'
                             % (type(_e).__name__, str(_e)[:200],
                                str(function_list)[:300], _tb.format_exc()))
            except OSError:
                pass
            return (OrderedDict(), 0, "")
        result= (gepResult, score, pickle_name)
        return result

    def clean_item_name (self,name):
        new_name = self.model.remove_prefix(name)
        new_name = self.model.remove_suffix(new_name)
        return new_name

    def original_arity(self, func):
        arity = None
        description = self.retrieve_ontology_item(func)
        #print('func')
        #print(func)
        if description and "_args" in description:
            arity = len(description["_args"])
        #if description and  "_args" not in description:
            #print ('description')
            #print (description)
        return arity

    def arity(self, func):
        arity = None
        new_func = self.clean_item_name(func)
        arity = self.model.emergent_functions_arity[new_func] \
            if new_func in self.model.emergent_functions_arity \
            else self.original_arity(new_func)
        return arity



    def function_list_arity(self, function_list_dont_modify):
        # THE BUG, one line, and it is April 4's (her call: "lets just
        # start with apr 4 and debug it for arity in general", after "if
        # there was a bug it would be minimal" - it was).
        #
        # The shortfall at each level was ASSIGNED rather than
        # accumulated, so a deficit followed by another level was erased.
        # It survived only when the list ended immediately. Measured
        # against the truth on seven shapes, April 4 is wrong on every
        # PARTLY FED node and right everywhere else:
        #     three inputs, one fed  -> reported 0, is 2
        #     two inputs, one fed    -> reported 0, is 1
        #     three inputs, two fed  -> reported 0, is 1
        # and correct on: none fed, fully fed, and the author's own
        # worked example in the gep() comment, which reproduces exactly.
        #
        # It does double damage, which is the whole story of today. The
        # returned number is stored as the emergent function's ARITY, so
        # a part short by two is recorded as needing nothing: a buyer
        # splices it in, supplies nothing, and the engine then calls a
        # three-input node with one argument. That is the starvation,
        # the wasted auditions, and a None cached against a chain that
        # was never broken.
        function_list = []
        function_list.extend(function_list_dont_modify)
        arity = 0
        deficit = 0
        if function_list:
            levels = OrderedDict([(1, [function_list.pop(0)])])
            current_level = 1
            more = True
            while more:
                length_next_level = 0
                for func in levels[current_level]:
                    arity = self.arity(func)
                    if arity is None:
                        # Unresolvable token in the chain — happens when an
                        # agent's decoded trade plan contains a placeholder
                        # (e.g. bare 'clusterer' / 'vectorSpace') that the
                        # post-2026-05-21 scaffolding-gated _resolve_placeholder
                        # declined to substitute. Per the "fail gracefully,
                        # don't silently default" directive: return None so
                        # obtain_trade_list can mark this chain as unevaluable
                        # and pass_all_tests rejects the offer cleanly (the
                        # `if itemlist:` guard treats falsy/None as failure).
                        return None
                    length_next_level += arity
                current_level += 1
                levels[current_level] = function_list[0:length_next_level]
                function_list = function_list[length_next_level:]
                deficit += length_next_level - len(levels[current_level])
                if not length_next_level:
                    break
                more = function_list or levels[current_level]
                #more = length_next_level

        return deficit

    def next_call_number_prefix(self):
        prefix = 'f' + str(self.model.emergent_functions_call_number) + '_'
        self.model.emergent_functions_call_number += 1
        return prefix



    def prefix_call_numbers(self,function_list):
        prefix_call_numbers = []
        functions_once = OrderedDict()
        for function in function_list :
            if function not in prefix_call_numbers:
                prefix = self.next_call_number_prefix()
                functions_once[function]= prefix+function

            prefix_call_numbers.append(functions_once[function])
        return prefix_call_numbers



    def gep(self, function_list_dont_modify):

        # assign input and output functions as defined by the Karva notation.
        # get arity of the items and divide the levels according to that arity,
        # then make the assignments across the levels

        # example.  for the following program list with the following arity, the karva notation result is the following
        # self.learnedProgram = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s']
        # self.arity = {'a':2,'b':3,'c':2,'d':2,'e':1,'f':1,'g':2,'h':1,'i':1,'j':1,'k':0,'l':0,'m':1,'n':1,'o':0,'p':0,'q':0,'r':0,'s':0}
        # self.results = {'a':['b','c'],'b':['d','e','f'],'c':['g','h'], 'd':['i','j'],'e':['k'],
        #      'f':['l'],'g':['m','n'],'h':['o'],'i':['p'],'j':['q'],'m':['r'], 'n':['s']}

        # divide into levels

        #function_list = []
        #function_list.extend(function_list_dont_modify)
        gep_result = OrderedDict()
        function_list = self.prefix_call_numbers(function_list_dont_modify)
        if function_list:
            levels = OrderedDict([(1, [function_list.pop(0)])])

            current_level = 1
            while function_list:
                length_next_level = 0
                for func in levels[current_level]:
                    noprefix = self.model.remove_prefix(func)
                    arity = self.arity(noprefix)
                    # An unresolvable token counts as taking nothing.
                    # It used to crash here - `int += None` - and one bad
                    # chain killed an entire run, in a market whose whole
                    # design expects broken chains to fail quietly and
                    # score zero. Never again from this line.
                    length_next_level += arity or 0
                current_level += 1
                levels[current_level] = function_list[0:length_next_level]
                function_list = function_list[length_next_level:]
                if not length_next_level:
                    break

            # make assignments


            for level, function_list in levels.items():
                next_level = level + 1
                cursor = 0

                for func in function_list:
                    noprefix = self.model.remove_prefix(func)
                    arity = self.arity(noprefix)
                    # Some chain entries (notably bare category prefixes like
                    # "preprocessor" that survived obtain_trade_list with
                    # unresolved upstream buys) return None arity. Treating
                    # them as 0-arity stubs keeps the chain assembly going
                    # rather than crashing the whole sim. The downstream test
                    # will get a partial chain and either pass or fail
                    # gracefully — the same handling as for any chain that
                    # produces None at a layer.
                    if arity is None:
                        arity = 0
                    next_cursor = cursor + arity
                    if next_level in levels:
                        gep_result[func] = levels[next_level][cursor:next_cursor]
                    else:
                        gep_result[func] =[]
                    cursor = next_cursor


        return gep_result

    def get_all_emergent_subroutines(self, function_list):
        # if any function in the list is emergent, then add it and its own emergent
        # subroutines to the list

        emergent_subroutines = set()

        for function in function_list:
            if function in self.model.emergent_functions:
                emergent_subroutines.update(self.get_all_emergents_set(function))
        return list(emergent_subroutines)

    def get_all_emergents_set(self,function_name, call_depth = 0):
        emergent_subroutines = set()
        if call_depth < self.p["recursive_trade_depth_limit"]:
            if function_name in self.model.emergent_functions:
                children = self.model.emergent_functions[function_name]
                for child in children:
                    level = call_depth+1
                    descendants = self.get_all_emergents_set(child, call_depth = level)
                    emergent_subroutines.update(descendants)
                emergent_subroutines.add(function_name)
            #print('emergent_subroutines')
            #print(emergent_subroutines)
        return emergent_subroutines

    def gep_clean(self, gepResult):
        #return true if this gepResult has an emergent function anywhere
        geptuple = self.geptuple(gepResult)
        clean = not any((x and self.emergent_pattern.match(x)) for x in geptuple)
        return clean

    def modular_gep(self, function_list):
        # take a function list, that has the emergent functions in it.  go through
        #  the list and for each emergent function it has, get its list,and for each
        # they have, get its list, until you have the original function list and
        # the entire tree of its subroutines.
        # now, send each list to gep and get a list of gep results.  send that list to
        #  another routine, combine_modules, that takes the list of gep results, and
        # creates a gep result that has only registry functions in it.

        #print('function_list')
        #print(function_list)
        emergent_functions = self.get_all_emergent_subroutines(function_list)

        #print('emergent_functions')
        #print(emergent_functions)
        gep_ordered_dict = OrderedDict()
        gep_ordered_dict['root'] = self.gep(function_list)
        gep_dict = OrderedDict([(f,self.gep(self.model.emergent_functions[f])) for f in emergent_functions])
        gep_ordered_dict.update(gep_dict)
        gep_result = self.combine_modules(gep_ordered_dict)

        #print ('gep_result')
        #print (gep_result)

        if not self.gep_clean(gep_result):
            print ('recursive function not allowed :')
            print(gep_result)
            gep_result = OrderedDict()
        return gep_result

    def make_equivalent_gep(self, gep_result):

        prefix_map = OrderedDict()
        equivalent_gep = OrderedDict()
        for func_name, arglist in gep_result.items():
            prefix = self.model.get_call_prefix(func_name)
            if prefix not in prefix_map:
                prefix_map[prefix ]= self.next_call_number_prefix()
            for arg in arglist:
                prefix = self.model.get_call_prefix(arg)
                if prefix not in prefix_map:
                    prefix_map[prefix ]= self.next_call_number_prefix()


        for func_name, arglist in gep_result.items():

            new_arglist = []
            for arg in arglist:
                prefix = self.model.get_call_prefix(arg)
                if prefix and (prefix in prefix_map):
                    new_arglist.append (prefix_map[prefix]+ self.model.remove_prefix(arg))
                else:
                    print('null prefix')
            prefix = self.model.get_call_prefix(func_name)
            new_func_name = prefix_map[prefix]+ self.model.remove_prefix(func_name)
            equivalent_gep[new_func_name]= new_arglist

        return equivalent_gep

    def find_unbounds(self,gep_result):
        unbounds = set()
        for func,arglist in gep_result.items():
            regular_function_arity = self.arity(func)
            if regular_function_arity is not None:
                if regular_function_arity > 0 and len(gep_result[func]) == 0:
                    unbounds.add(func)
            #elif func in self.model.emergent_functions_arity and self.model.emergent_functions_arity[func] > 0 \
                    #and len(gep_result[func]) == 0:
                #unbounds.add(func)
            for arg in arglist:
                regular_function_arity = self.arity(arg)
                if regular_function_arity is not None:
                    if regular_function_arity > 0 and (arg not in gep_result or len(gep_result[arg]) ==0):
                        unbounds.add(arg)
                #elif arg in self.model.emergent_functions_arity and self.model.emergent_functions_arity[arg]> 0 \
                        #and arg not in gep_result or len(gep_result[arg]) ==0:
                    #unbounds.add(arg)
        return list(unbounds)

    def flattern(self, A):
        rt = []
        for i in A:
            if isinstance(i, list):
                rt.extend(self.flattern(i))
            else:
                rt.append(i)
        return rt


    def geptuple(self, gepresult):
        # flatten, remove prefix, make a tuple
        geptuple = tuple()
        if (len(gepresult)):
            alist = self.flattern(gepresult.values())
            alist.extend(gepresult.keys())
            blist = [self.model.remove_prefix(x) for x in alist]
            geptuple = tuple(set(blist))
        return (geptuple)

    def combine_modules(self,emergent_function_dict_dont_modify):

        #print ('emergent_function_dict_dont_modify')
        #print(emergent_function_dict_dont_modify)
        emergent_function_dict = copy.deepcopy(emergent_function_dict_dont_modify)
        # take the list of gep results, and
        # create a gep result that has only registry functions in it.

        #  for every function that uses emergent functions Q,in dictionary of all emergent functions as gep results D
        # take each emergent separately.  the emergent functions that have arity will appear as a key.
        # remove them from the list and set aside  a, b, cOrderedDict([('root', OrderedDict([('f0_test_clusterer_silhouette', ['f1_clusterer_sklearn_affinityPropagation_10clusters']), ('f1_clusterer_sklearn_affinityPropagation_10clusters', ['f2_vectorSpace_gensim_doc2vec_50size_200iterations_5minFreq']), ('f2_vectorSpace_gensim_doc2vec_50size_200iterations_5minFreq', ['f3_preprocessor_freetext_tag']), ('f3_preprocessor_freetext_tag', ['f4_preprocessor_freetext_shuffle_stochastic4']), ('f4_preprocessor_freetext_shuffle_stochastic4', ['f5_data_freetext_internetResearchAgency'])]))])
        # for each of these usages, create a new copy of the emergent function that has different call order numbers (but arranged the same) d, e, f
        # in d e f, change the call number of the root, if there is one,  to be the call number from the removed functions, and change every place that the emergent function is in an input list to its new name (a b c)
        # then, in d e f,  map functions that are missing inputs to the inputs of the removed functions(a b c)
        # put contents of modified emergent functions d e f  back into the original function list Q
        # remove the emergent function from the larger list D, emergent_function_dict
        #OrderedDict([('root', OrderedDict([('f0_test_clusterer_silhouette', ['f1_clusterer_sklearn_affinityPropagation_10clusters']), ('f1_clusterer_sklearn_affinityPropagation_10clusters', ['f2_vectorSpace_gensim_doc2vec_50size_200iterations_5minFreq']), ('f2_vectorSpace_gensim_doc2vec_50size_200iterations_5minFreq', ['f3_preprocessor_freetext_tag']), ('f3_preprocessor_freetext_tag', ['f4_preprocessor_freetext_shuffle_stochastic4']), ('f4_preprocessor_freetext_shuffle_stochastic4', ['f5_data_freetext_internetResearchAgency'])]))])

        #all of these are emergent functions, but some of them dont use other emergent functions and some do

        use_emergent_functions = OrderedDict(
            [(fname, gep_result) for fname, gep_result in emergent_function_dict.items() if \
             not self.gep_clean(gep_result)])
        previous_length = sys.maxsize
        while (not len(use_emergent_functions) == 0) and len(use_emergent_functions)< previous_length :
            previous_length = len(use_emergent_functions)

            use_only_non_emergent_functions = OrderedDict(
                [(fname, gep_result) for fname, gep_result in emergent_function_dict.items() \
                    if self.gep_clean(gep_result)])
            # we are depending on emergent_function_dict contents to be passed by value
            # for every function that uses emergent functions Q (user_gep_result),
            # take each emergent separately.  the emergent functions that have arity will appear as a key.
            # remove them from the list and set aside  a, b, c (emergent_funct_usages)

            for user_name , user_gep_result in use_emergent_functions.items():
                for non_user_name, non_user_gep_result in use_only_non_emergent_functions.items():
                    #emergent_funct_usages = {name: gep_results for name, gep_results in
                                             #use_emergent_functions.items() \
                                             #if non_user_name in name}
                    emergent_funct_usages = OrderedDict([(name, gep_results) for name, gep_results in \
                                             user_gep_result.items() if non_user_name in name])

            #for each of these usages, create a new copy of the emergent function non_user_gep_result that has
            # different call order numbers (but arranged the same) d, e, f (combined functions)

                    combined_functions = [self.make_equivalent_gep (non_user_gep_result)for _ in emergent_funct_usages]


            # in d e f, (combined functions) change the call number of each root key
            # to be the call number from the removed functions, and change every place that the
            # emergent function is in an input list to its new name (a b c)(emergent_funct_usages)

                    for i, (name, arg_list) in enumerate(emergent_funct_usages.items()):
                        prefix = self.model.get_call_prefix(name)
                        # for gep_result in combined_functions:
                        gep_result = combined_functions[i]
                        if (len(gep_result)):
                            root = next(iter(gep_result.items()))[0]
                            new_name = prefix + self.model.remove_prefix(root)
                            gep_result[new_name] = gep_result.pop(root)
                            gep_result.move_to_end(new_name, last=False)
                            for fname, arglist in user_gep_result.items():
                                if name in arglist:
                                    arglist[arglist.index(name)] = new_name

                            # then, in d e f, (combined functions)  map functions that are missing inputs to the inputs
                            # of the removed functions(a b c) (emergent_funct_usages)
                            # match the prefix of the root in combined functions to the prefix of the key in emergent funct usages

                            input_assignments = OrderedDict()
                            unbound_list = self.find_unbounds(gep_result)

                            cursor = 0
                            for unbound in unbound_list:
                                # arity = self.model.emergent_functions_arity[unbound]
                                arity = self.arity(unbound)
                                next_cursor = cursor + arity
                                input_assignments[unbound] = arg_list[cursor:next_cursor]
                                cursor = next_cursor
                            # put contents of modified emergent functions d e f  back into the original function list Q (user_gep_result)

                            user_gep_result.update(gep_result)
                            if list(user_gep_result.keys()).index(name)== 0:  #this is the root
                                user_gep_result.move_to_end(new_name, last=False)

                            user_gep_result.update(input_assignments)
                            user_gep_result.pop(name)
                        else:  # empty function
                            user_gep_result.pop(name)
                            for fname, arglist in user_gep_result.items():
                                if name in arglist:
                                    arglist[arglist.index(name)] = None

        # remove the emergent function from the original function list D (emergent_function_dict)
                        ename = self.model.remove_prefix(name)
                        if ename in emergent_function_dict:
                            emergent_function_dict.pop(ename)

            use_emergent_functions = OrderedDict(
                [(fname, gep_result) for fname, gep_result in emergent_function_dict.items() \
                    if not self.gep_clean(gep_result)])

            #use_emergent_functions = {fname: gep_result for fname, gep_result in emergent_function_dict.items() \
		    #if any ( f in tuple(self.model.emergent_functions.keys()) for f in  tuple(gep_result.values()))}

        #print ('emergent_function_dict')
        #print(emergent_function_dict)
        result = emergent_function_dict['root']
        return (result)


    def transfer_funds(self, buyer, buynum, seller, sellnum, price):
        buyer.wealth -= price
        seller.wealth += price
        buyer.payment_notification(-price, buynum)
        seller.payment_notification(price, sellnum)
        # Payment ledger (2026-07-19, flag-gated): transfer_funds is the one
        # choke point every settlement AND every cascade sub-payment passes
        # through, so this row-per-payment file is the ground truth for
        # who-paid-whom — the instrument the hiring question needs.
        if not os.environ.get('MARKETSIM_WARM_CHILD'):  # payment ledger is law,
            # not a switch (the curator, 2026-08-06); warm children write no records
            try:
                import os as _os
                _path = self.model.parameters['output_path'] + 'payments.csv'
                # Writer-side rotation (the curator, 2026-08-05): a restart
                # must never append to a previous run's ledger. Once per
                # process, an existing file is MOVED to a timestamped
                # archive name (never deleted) before the first write.
                if not getattr(self.model, '_payments_rotated', False):
                    self.model._payments_rotated = True
                    if _os.path.exists(_path):
                        import time as _time
                        _os.rename(_path, _path.replace(
                            'payments.csv',
                            'payments_arch_%s.csv' % _time.strftime('%H%M%S')))
                _new = not _os.path.exists(_path)
                with open(_path, 'a') as _fh:
                    if _new:
                        _fh.write('time;buyer;seller;price\n')
                    _fh.write('%s;%s;%s;%.6f\n' % (self.model.schedule.time,
                              buyer.unique_id, seller.unique_id, price))
            except Exception:
                pass

    def distribute_funds(self, buy, buynum, call_depth=0):
        # Mirrors the depth limit in obtain_trade_list. Without this, cyclic
        # sub-buy chains (possible when sellers can both buy and sell the same
        # item type, e.g. universal trade plans) cause infinite recursion in
        # the payment distribution path.
        if call_depth > self.p.get("recursive_trade_depth_limit", 10):
            return

        offer = buy['offers'][buy['chosen']]
        sellnum = offer['trades'][0]

        sells = self.b[offer['agent']]['trades'][offer['trades'][0]:]

        # Mirror the section-boundary fix from obtain_trade_list: the chain
        # ends at the next stop OR the next sell entry, whichever comes first,
        # so multi-role agents' payment cascades stay within the matched
        # role's section.
        found = False
        stopval = -1
        for i, trade in enumerate(sells):
            if i == 0:
                continue
            if (not found) and (
                trade['item'].startswith('stop')
                or trade['type'] == 'stop'
                or trade['type'] == 'sell'
            ):
                found = True
                stopval = i

        # Boundary fix (2026-07-19): stopval=-1 means NO boundary was found —
        # the section runs to the end of the list. The old unconditional slice
        # turned that into sells[0:-1], silently emptying single-sell sections
        # and skipping payment entirely (the dead-wealth root cause; sibling of
        # the chosen=0 bug fixed earlier in this same region).
        if found:
            sells = sells[0:stopval]
        # now find the buys and insert settled software (or ignore)
        # list.insert(index, elem)

        if sells:
            self.transfer_funds(self, buynum, self.model.schedule.agents[offer['agent']], sellnum, buy['price'])
        # BUG FIX: the old truthy check `sell['chosen']` silently dropped
        # sub-buys where chosen=0 (a valid first-offer index). This broke the
        # payment chain for two-thirds of middleman trades in oneofeach and is
        # why specialists never got paid. Mirrors the correct check in
        # obtain_trade_list.
        buylist = [i for i, sell in enumerate(sells)
                   if sell['type'] == 'buy'
                   and sell.get('chosen') is not None]

        for i in buylist:
            sellnum = offer['trades'][0] + i
            seller = self.model.schedule.agents[offer['agent']]
            seller.distribute_funds(sells[i], sellnum, call_depth + 1)


    def obtain_trade_list(self, offer, call_depth = 0):
        # Chain construction for the offered item. Scans the seller's
        # trades list starting at the offered sell, terminating at:
        #   - the next stop codon, OR
        #   - the next sell entry (boundary of a different role's section).
        # The next-sell terminator addresses the prior "todo: implement more
        # than one item sold by agent" by treating each sell as the start of
        # its own chain. Without this, a multi-role agent's chain for sell:X
        # would absorb subsequent sell:Y as a fake component, producing
        # malformed pipelines that fail in execution.

        if call_depth > self.p["recursive_trade_depth_limit"]:
            unique_id = None
        else:
            unique_id = "f" + str(self.model.schedule.time) + "_" + str(self.unique_id) \
                        + "_" + str(offer['agent']) + "_" + str(offer['trades'][0])
            if unique_id not in self.model.emergent_functions:
                # print("in obtain software")
                sells = self.b[offer['agent']]['trades'][offer['trades'][0]:]
                # print("sells")
                # print(sells)
                found = False
                stopval = -1
                for i, trade in enumerate(sells):
                    # Skip the offered sell itself at index 0.
                    if i == 0:
                        continue
                    if (not found) and (
                        trade['item'].startswith('stop')
                        or trade['type'] == 'stop'
                        or trade['type'] == 'sell'
                    ):
                        found = True
                        stopval = i

                # F2 (2026-09-10): stopval=-1 means NO boundary was found and
                # the section runs to the end of the plan; an unguarded slice
                # made that sells[0:-1] and dropped the last ingredient - the
                # bug distribute_funds fixed on 2026-07-19. Same guard here.
                if found:
                    sells = sells[0:stopval]

                #We want to refer to this piece of code again in the same iteration without making an
                # extra copy of it in the emergent_functions routine.  to do so give it a unique id.
                # uniqueid is made of self.model.schedule.time, self.unique_id, offer['agent'], offer['trades'][0]


                # now find the buys and insert settled software (or ignore)

                #buys_inserted = []
                buylist = [i for i, sell in enumerate(sells) if sell['type'] == 'buy' \
                           and 'chosen' in sell and sell['chosen'] is not None]
                # Diagnostic (her arranging-rounds debug 2026-08-20):
                # count section input-buys seen SETTLED vs UNSETTLED at
                # assembly time - the unsettled ones degrade to bare
                # family names and vanish from the delivered chain.
                if self.p.get('pass_diag'):
                    for _i7, _s7 in enumerate(sells):
                        if _s7.get('type') == 'buy':
                            self._pass_diag(
                                'input-%s;%s' % (
                                    'settled' if _i7 in buylist
                                    else 'UNSETTLED',
                                    stop_cut(str(_s7.get('item', '')))))

                # this is the straight up insertion of code, rather than a subroutine
                # cursor = 0
                # if not buylist:
                #     buys_inserted = sells
                # for i in buylist: #iterate by writing out all the sells, and then the buy insertion
                #     next_cursor = i
                #     buys_inserted.extend(sells[cursor:next_cursor])
                #     cursor = next_cursor + 1 #skip the actual buy
                #     #sell = buylist[i]
                #     sell = sells[i]
                #     buys_inserted.extend(self.obtain_trade_list(sell['offers'][sell['chosen']]))

                clean_funcs = []
                for _i8, sell in enumerate(sells):
                    func = sell['item']
                    clean_func = [stop_cut(func)]
                    if sell.get('type') == 'buy' and _i8 not in buylist:
                        # A buy row is a REQUEST, not a construct. Until it is
                        # settled the seller does not have the part, so it
                        # contributes nothing to the chain. Found 3 Oct 2026
                        # (night build): with the item kept, an unsettled buy
                        # for the right decoder leaf served as a free part -
                        # 139 of 400 paid cipher products carried unsettled
                        # buys, and composition was free. Her 21 Aug ruling
                        # (an unfilled input cannot sell) said as much.
                        clean_funcs.append('')
                        continue
                    clean_funcs.append(clean_func[0])


                for i in buylist:
                    sell = sells[i]
                    depth = call_depth +1
                    clean_funcs[i]=  self.obtain_trade_list(sell['offers'][sell['chosen']], call_depth = depth)

                # For each chain item: if it's directly in the registry
                # use it as-is; else it is an unresolved placeholder and
                # _resolve_placeholder returns None (defaults are never
                # substituted — the curator's 2026-07-04 ruling). The item is
                # dropped; the malformed chain fails at runtime, the
                # exception is caught by MarketSim.memoise_pickle and
                # logged to caught_exceptions.log, score=0, no trade.
                # Graceful failure with diagnostic visibility.
                resolved = []
                for program in clean_funcs:
                    if not program:
                        continue
                    if (self.model.remove_suffix(program) in self.model.registry
                            or program in self.model.emergent_functions):
                        resolved.append(program)
                        continue
                    # HER LAW (2026-08-19 night): "at the function
                    # level a stop always means use the defaults of
                    # the function" - no flag, no option. Mechanical
                    # form: cut the name at its stop codon; if the cut
                    # lands EXACTLY on a registry key, that key IS the
                    # function's library-default variant and the
                    # element resolves to it. A family-level cut
                    # ('clusterer') is not a registry key, so her
                    # standing "no default at family level" ruling
                    # holds by construction. (The old scaffolding
                    # option poor_default_resolution is dead: defaults
                    # at function level are LAW, not an option.)
                    _dcut = stop_cut(program)
                    if _dcut != program and _dcut in self.model.registry:
                        resolved.append(_dcut)
                        continue
                only_existing = resolved


                computed_arity = self.function_list_arity(only_existing)
                # A POSITIVE DEFICIT IS NOT AN ERROR - IT IS A PART.
                #
                # I briefly rejected chains whose expression did not close,
                # and her warning caught it: "if there is a buy instead of
                # a construct, it should just continue the expression with
                # the next agent's code." That is exactly what the deficit
                # carries. function_list_arity returns children demanded
                # minus children supplied, and that number is stored as the
                # emergent function's ARITY - so when a buyer splices a
                # purchased subtree in, arity() reports the seller's unmet
                # inputs and the buyer's following plans supply them.
                #
                # Rejecting a positive deficit therefore forbids selling a
                # part that needs an input, which is the whole substance of
                # this market. Reverted. The guard against evaluating an
                # unfinished expression belongs where it is EVALUATED, not
                # where it is assembled.
                if computed_arity is None:
                    # Chain has an unresolvable placeholder (e.g. bare
                    # 'clusterer' that didn't get a brand-aware default).
                    # Mark the chain empty so pass_all_tests rejects the
                    # offer via its `if itemlist:` guard — no silent default,
                    # no crash. See function_list_arity comment + the
                    # 2026-05-21 "fail gracefully" directive.
                    self.model.emergent_functions[unique_id] = []
                    self.model.emergent_functions_arity[unique_id] = 0
                else:
                    self.model.emergent_functions[unique_id] = only_existing
                    self.model.emergent_functions_arity[unique_id] = computed_arity

        import os as _os
        if _os.environ.get('MARKETSIM_DEBUG_ASSEMBLY'):
            print(f"[ASSEMBLY] depth={call_depth} id={unique_id} stored={self.model.emergent_functions.get(unique_id)}")
        return unique_id #self.model.emergent_functions[unique_id]

    def pass_all_tests(self, buy, offernum):

        # Implements a testing system where a human or agent can require any test in a category on any data in a
        # category if generality is required, or all tests on data that are listed. This function tells if the agent
        # has passed all tests, and saves the scores for feedback to the agent. One test that must be passed is the
        # offerer must have the item.  It does not have to be all bought, but the part that is said to be the item
        # must be present, if the offer is the result of a sell that was not constructed,but bought.  Second,
        # every test listed in the tests, before the first stop codon, must be passed The test score on each test is
        # noted in the test score description

        # First retrieve item to buy.  Item is defined as including all from the sell statement to the next stop or the
        # end of the list.  if there are buys, and they are settled, retrieve the bought software

        # a stop in the first place of either the test or the data is a stop for all tests.
        # if the stop is midway through the test or the data, it indicates general tests or data,
        # that is, if it passes for any test or data in this category over this threshold it passes
        cumulative_score = 0
        pickle_name = ""
        gepResult = None
        pass_all_tests = True
        # HER LAW (2026-08-20): every test on a buy is of the
        # want's own product type, so every live test must be
        # RUNNABLE against the offered item. type_ok records that
        # and NOTHING else - it is deliberately kept apart from
        # the score threshold, because a bar of zero waives the
        # SCORE and must never waive the TYPE. Conflating the two
        # is what let a classifier buyer pay for a vectorSpace
        # encoder (board 9725, slots 28/29/30).
        type_ok = True
        itemlist = None
        numtests = 0
        if self.model.parameters.get('judged_per_iteration'):
            # One judgment per iteration (the curator, 2026-08-06): only the
            # want holding this step's single testing slot may run
            # tests; a want with runnable tests that lacks the slot
            # cannot settle this step. Wants with no active tests trade
            # token-only as ever, so the bootstrap lane is untouched.
            _active = False
            for _td in buy.get('tests', []):
                if _td.get('stophere'):
                    break
                _tt = stop_cut((_td.get('test') or ''))
                _dd = stop_cut((_td.get('data') or ''))
                if not _tt or not _dd:
                    break
                _active = True
                break
            if _active and getattr(self.model, '_judged_key', None) != \
                    (self.unique_id, stop_cut(str(buy.get('item', '')))):
                return (False, None, 0, "")
        if offernum is not None:
            # HER LAW (2026-08-18, closing the price-band
            # contradiction; measured 25% of settles rode the
            # fallback): "of course the settle must have overlapping
            # bands on blind and all the other runs." A candidate
            # whose price band does not overlap the buy's cannot
            # settle, on any path (best-of-k, walk, closest,
            # credulous) - and is refused here BEFORE its programs
            # execute. The midpoint-average fallback in price() is
            # thereby dead code for settles; no flag, as with every
            # market law.
            try:
                _off_b = buy['offers'][offernum]
                _sid_b = int(_off_b.get('agent'))
                _sm_b = self.model.blackboard[_sid_b]
                _sell_b = next((t for t in _sm_b.get('trades', [])
                                if t.get('type') == 'sell'), None)
            except (TypeError, ValueError, IndexError, KeyError):
                _sell_b = None
            if (_sell_b is not None and 'midpoint' in buy
                    and 'midpoint' in _sell_b
                    and not self.p.get('price_list')):
                # Under THE PRICE LIST (her ruling 2026-10-01: price set
                # aside, the learned price floats stay in the genome
                # UNREAD) no band can refuse an audition; the band gate
                # returns with price, decided at the time. Found 2 Oct
                # night: 1,378 band refusals in 16 boards of S4's first
                # launch against 247 settlements.
                _bl = max(buy['midpoint'] - buy.get('range', 0.0), 0.0)
                _bh = buy['midpoint'] + buy.get('range', 0.0)
                _sl = max(_sell_b['midpoint']
                          - _sell_b.get('range', 0.0), 0.0)
                _sh = _sell_b['midpoint'] + _sell_b.get('range', 0.0)
                if min(_bh, _sh) < max(_bl, _sl):
                    self._pass_diag('band')
                    return (False, None, 0, "")
            func_name = self.obtain_trade_list(buy['offers'][offernum])
            itemlist = self.model.emergent_functions[func_name]
            if itemlist:
                stop_codon_reached = False

                for test_dict in buy['tests']:
                    if test_dict['stophere']:
                        stop_codon_reached = True
                    if not stop_codon_reached:
                        stop_cut_off = [stop_cut(test_dict['test'])]
                        clean_test = stop_cut_off[0]
                        stop_cut_off = [stop_cut(test_dict['data'])]
                        clean_data = stop_cut_off[0]
                        if not clean_test or not clean_data:
                            stop_codon_reached = True
                        else:
                            # The data or tests may be general categories, so see if there are tests
                            # or data in those categories
                            testlist = self.descendants(clean_test)
                            if not testlist:
                                testlist.append(clean_test)
                            datalist = self.descendants(clean_data)
                            if not datalist:
                                datalist.append(clean_data)


                            anypass = False
                            entry_ran = False
                            for test in testlist:
                                for data in datalist:
                                    # if any is passed in this group, then give a pass
                                    #however, do not run it unless the software is the same type as the test
                                    item_type = self.item_type(itemlist[0])
                                    test_item_type = self.test_item_type (test)
                                    if item_type  and (item_type == test_item_type) and len(self.retrieve_ontology_item(test)):
                                        # The buyer's test supplies the corpus: one
                                        # data terminal per UNFILLED input slot of
                                        # the offered item chain. Linear chains have
                                        # arity 1, so this is the historical single
                                        # append; a composed product carrying k
                                        # purchased sub-chains has k empty corpus
                                        # slots and starves past the first without
                                        # this (found 2026-07-06: composed consensus
                                        # offers truncated at the second sub-chain).
                                        #
                                        # HER RULING (2026-08-12, option 1): REFUSE,
                                        # DO NOT GUESS. A fallback of one input was
                                        # safe only while every product took one
                                        # input - wave 16's catalogue was 234 of 261
                                        # single-input. With the branch-capable
                                        # catalogue, 47 of 261 listings take two or
                                        # three, and guessing one hands a three-input
                                        # product a single argument. That raises, the
                                        # audition ends, no pickle name is produced,
                                        # and nothing is delivered: both v27 runs
                                        # recorded ZERO deliveries this way, with
                                        # 3,546 starved calls in a 400,000-line
                                        # sample of one exception log.
                                        #
                                        # The chain is filtered FIRST so the
                                        # requirement is counted on the program that
                                        # will actually run. The old order counted it
                                        # on the unfiltered list and dropped entries
                                        # afterwards, so a parent could silently lose
                                        # a child after its slots had been counted.
                                        _chain = [p for p in itemlist
                                                  if (self.model.remove_suffix(p) in self.model.registry
                                                      or p in self.model.emergent_functions)]
                                        if not _chain:
                                            continue
                                        item_arity = self.function_list_arity(_chain)
                                        if item_arity is None:
                                            continue
                                        if item_arity < 1:
                                            item_arity = 1
                                        program_list = [test]
                                        program_list.extend(_chain)
                                        # A JOINING HEAD TAKES ITS OPERANDS
                                        # FROM THE TAIL. A plain corpus needs
                                        # none and the tail stays unexpressed.
                                        _need = self.arity(data) or 0
                                        _dx = [data]
                                        if _need:
                                            _tl = test_dict.get('data_tail') or []
                                            if len(_tl) >= _need:
                                                _dx = [data] + list(_tl[:_need])
                                        for _ in range(item_arity):
                                            program_list.extend(_dx)

                                        #non-coding segments are implemented when non completed functions are ignored
                                        #so dont put non completed funcitons in the registry.
                                        program_list = [program for program in program_list \
                                                        if (self.model.remove_suffix(program) in self.model.registry or \
                                                            program in self.model.emergent_functions)]

                                        # Shared-scenario port hook (the curator
                                        # 2026-07-16): publish THIS
                                        # audition's buyer data terminal so
                                        # the port want-scorers bind their
                                        # scoring truth to the buyer — the
                                        # LLM buyer scores deliveries
                                        # against its own test.csv, and a
                                        # self-contained foreign chain
                                        # must not be scored against the
                                        # truth it carries. Read only by
                                        # Registry.PORT_BUYER_KEYED
                                        # scorers; inert in every
                                        # historical scenario (none of
                                        # them contains such a scorer).
                                        from simulation import Registry as _PortRegistry
                                        _PortRegistry.PORT_BUYER_DATA = data
                                        # THE BUYER SUPPLIES THE TRUTH (flag
                                        # buyer_supplies_truth, default off).
                                        # Published beside the corpus so a test
                                        # needing ground truth can take the
                                        # buyer's rather than requiring the
                                        # seller to have guessed it.
                                        _PortRegistry.PORT_BUYER_TRUTH = bool(
                                            self.p.get('buyer_supplies_truth'))
                                        try:
                                            gepResult, score, pickle_name = self.perform_test(program_list)
                                        finally:
                                            _PortRegistry.PORT_BUYER_DATA = None
                                            _PortRegistry.PORT_BUYER_TRUTH = False
                                                            # A mis-shaped chain (e.g. a bad
                                        # arity-2 tree) can yield a non-scalar
                                        # "score"; that is not a score. Treat
                                        # it as a failed evaluation.
                                        if not isinstance(score, (int, float)):
                                            score = 0

                                        # EVERY SCORE IS ON [0, 1] (her ruling
                                        # 2026-08-20, "normalize"). Measured
                                        # before the fix: scores ran to 186.02
                                        # while the evolved bars are drawn from
                                        # max(0, 2f-1) and so live in [0, 1] -
                                        # the same bar therefore meant something
                                        # different depending on which test the
                                        # buy had drawn, and an unbounded test
                                        # cleared any bar for free. The offender
                                        # was test_clusterer_calinskiHarabaz,
                                        # which divides an UNBOUNDED index by a
                                        # fixed 200 and never clipped; that test
                                        # is fixed at source too. This clamp is
                                        # the root guard, so no future test can
                                        # put an out-of-range number into the bar
                                        # comparison, the buyer's aspiration or
                                        # human utility. NaN is not a score.
                                        score = float(score)
                                        if score != score:      # NaN
                                            score = 0.0
                                        elif score < 0.0:
                                            score = 0.0
                                        elif score > 1.0:
                                            score = 1.0

                                        # record the score no matter what, as feedback
                                        if 'results' not in test_dict:
                                            test_dict['results'] = []
                                       # result = {'offer': offernum, 'score': score, 'time': self.model.schedule.time,
                                                  #'test': test, 'data': data}
                                        result = OrderedDict([('offer', offernum), ('score', score), ('time', self.model.schedule.time),
                                                              ('test', test), ('data', data)])
                                        seller = self.model.schedule.agents[buy['offers'][offernum]["agent"]]
                                        tradenum = buy['offers'][offernum]["trades"][0]
                                        seller.seller_score_notification(score,tradenum)

                                        # PER-TEST ATTRIBUTION (her 2026-08-20
                                        # "3 yes"). The score0/tested-pass tokens
                                        # book final_score - the MEAN over every
                                        # test that ran for this buy - under the
                                        # buy's FIRST test name, so several
                                        # different test names showed an
                                        # identical maximum because they were the
                                        # same auditions. This token is emitted
                                        # once per test ACTUALLY RUN, naming that
                                        # test, its data, its own score and its
                                        # own bar. pass_diag worlds only.
                                        if self.p.get('pass_diag'):
                                            self._pass_diag(
                                                'ran;test=%s;data=%s;score=%.6f;bar=%.6f;item=%s'
                                                % (stop_cut(str(test)),
                                                   stop_cut(str(data)),
                                                   float(score),
                                                   float(test_dict.get('threshold', 0.0)),
                                                   stop_cut(str(itemlist[0]) if itemlist else '')))

                                        numtests += 1
                                        entry_ran = True
                                        cumulative_score += score
                                        test_dict['results'].append(result)

                                        # THE market law (2026-07-14, the curator's
                                        # missing-market fix): no static bar —
                                        # anything that runs and scores may
                                        # settle; the buyer's moving aspiration
                                        # in choose_partners decides when to
                                        # stop walking, and the score-scaled
                                        # price pays quality its premium. The
                                        # genome's threshold float stays in the
                                        # vector layout (decode compatibility)
                                        # but is deliberately unused. The old
                                        # fixed-bar law lives in git history
                                        # only, per the curator's no-flags ruling.
                                        if score is not None and score > \
                                                test_dict.get('threshold', 0.0):
                                            anypass = True
                                    else:
                                        pass_all_tests = False

                            if not entry_ran:
                                type_ok = False
                            if not anypass:
                                pass_all_tests = False

                    elif gepResult is None and itemlist:
                        gepResult = self.modular_gep(itemlist)
            else:
                pass_all_tests = False
        else:
            pass_all_tests = False

        final_score = cumulative_score / numtests if numtests else 0
        results = (pass_all_tests, gepResult, final_score, pickle_name)
        # Audition reason census (her settle-chain debug 2026-08-19):
        # config-flag instrumentation only - one reason token per call.
        if self.p.get('pass_diag'):
            if offernum is None:
                _rz = 'no-offer'
            elif not itemlist:
                _rz = 'noitem'
            else:
                _had_live = False
                for _td9 in buy.get('tests', []):
                    if _td9.get('stophere'):
                        break
                    if stop_cut(str(_td9.get('test', ''))) \
                            and stop_cut(str(_td9.get('data', ''))):
                        _had_live = True
                    break
                if not _had_live:
                    _rz = 'notest-pass' if pass_all_tests else 'notest-FAIL'
                elif numtests == 0:
                    # SAY WHAT WAS COMPARED. 'typemiss' alone cannot
                    # distinguish 'the seller really is the wrong kind'
                    # from 'the assembled chain no longer looks like what
                    # the seller advertised'. The audition compares the
                    # family of itemlist[0] - the FIRST SURVIVING element
                    # of the assembled chain, after obtain_trade_list has
                    # replaced bought entries and only_existing has
                    # dropped whatever is not in the registry - against
                    # the family of the buyer's test. Record both, plus
                    # the advertised sell item, so the three can be
                    # compared directly. pass_diag worlds only.
                    _il0 = ''
                    try:
                        _il0 = str(itemlist[0]) if itemlist else ''
                    except (TypeError, IndexError):
                        _il0 = ''
                    _tf9 = ''
                    for _td9 in buy.get('tests', []):
                        if _td9.get('stophere'):
                            break
                        _tf9 = self.test_item_type(
                            stop_cut(str(_td9.get('test', '')))) or ''
                        break
                    _adv9 = ''
                    try:
                        _o9 = buy['offers'][offernum]
                        _s9 = self.b[_o9['agent']]['trades'][_o9['trades'][0]]
                        _adv9 = stop_cut(str(_s9.get('item', '')))
                    except (TypeError, KeyError, IndexError):
                        _adv9 = ''
                    _rz = 'typemiss;chainhead=%s;chainfam=%s;testfam=%s;advertised=%s;len=%d' % (
                        _il0, self.item_type(_il0) or '', _tf9, _adv9,
                        len(itemlist) if itemlist else 0)
                else:
                    _tn9 = ''
                    for _td9 in buy.get('tests', []):
                        if _td9.get('stophere'):
                            break
                        _tn9 = stop_cut(str(_td9.get('test', '')))
                        break
                    _thr9 = 0.0
                    for _td9 in buy.get('tests', []):
                        if _td9.get('stophere'):
                            break
                        _thr9 = float(_td9.get('threshold', 0.0))
                        break
                    _rz = '%s;%.4f;%s;%s;bar=%.4f' % (
                        'tested-pass' if pass_all_tests else 'score0',
                        final_score, _tn9,
                        stop_cut(str(buy.get('item', ''))), _thr9)
            self._pass_diag(_rz)
        buy['_type_ok'] = 1 if type_ok else 0
        return results

    def _pass_diag(self, reason):
        """Append one audition-outcome token (pass_diag worlds only)."""
        if not self.p.get('pass_diag') or os.environ.get('MARKETSIM_WARM_CHILD'):
            return
        try:
            with open(os.path.join(
                    self.p.get('output_path', '.'), 'pass_diag.log'),
                    'a') as _f9:
                _f9.write(reason + '\n')
        except OSError:
            pass


    # AN UNRUNNABLE TEST IS NOT A TYPE VERDICT (flag
    # unrunnable_test_is_no_test, default off, 2026-08-21).
    #
    # The type gate reads `entry_ran` - did any test actually execute - and
    # calls a failure "the wrong kind of good". That conflates two different
    # things: the seller really being the wrong kind, and the BUYER'S OWN test
    # machinery never starting. Measured on gate_truth_seed306: of 47 attempts
    # by a classifier to buy a labeler, 44 carried the test `test_labeler` and
    # every one was refused with _type_ok=0, while the 3 carrying no test at
    # all settled. `labeler` is a product family with no test family in the
    # ontology, so its buys decode the bare category name, which resolves to
    # nothing. The families match perfectly - a labeler offered to a labeler
    # buyer - so nothing is the wrong kind; the test simply does not exist.
    #
    # That single refusal is what blocks multi-agent composition, because a
    # labeler is the part a classifier must buy for 21 of the 56 authored
    # humans to be satisfiable at all.
    #
    # The rule: if a buy's test names nothing real, the buy has no usable test
    # and falls into the untested lane - the same lane those 3 successful
    # purchases used. A test that DOES resolve and then fails still refuses,
    # so a genuinely wrong-kind seller is still caught.
    def test_names_something_real(self, testname):
        # NOT descendants(): it returns the name itself when a category has no
        # real children, so it never comes back empty and the check always
        # passed. Measured 2026-08-21: descendants('test_nonsense') == 1.
        # The ontology NODE is the honest discriminator - a real test carries
        # its metadata (_args, _weight, _return) and a name that exists only
        # as a decoded string carries nothing.
        t = stop_cut(str(testname or ''))
        if not t:
            return False
        try:
            return bool(len(self.retrieve_ontology_item(t)))
        except (AttributeError, KeyError, TypeError):
            return False

    def choose_partners(self):
        # for every buy trade, roll to pick who fills the slot. If a test is required, run what the agent has
        # through the test, and if it doesnt pass, nothing fills the slot this time.  If an agent buys goods from
        # an agent that does ot have them, the same thing happens. All other buys can not be redone once accepted.
        # Whatever the state of the called routine when the human accepts it is what is paid, as opposed to when the
        # buying agent accepts it. In other words, an agent can If the agent is a human, who accepts the trade,
        # money is disbursed throughout the network according to the chain of contracted prices.

        # HUMANS SETTLE LAST (her go, 2026-08-20 - the arranging fix
        # for THE ROOT BUG): an authored human's buys settle only in
        # the FINAL choose_partners round of the board. Learners settle
        # from round one and arrange all day; human demand arrives at
        # the market's close, when every supply chain the learners
        # planned is settled and assembles complete (the lazy recursive
        # assembly walks the live tree at the human's audit, so one
        # rule completes every depth). Restores the waiting that the
        # 2018 thresholds provided by failure, without failure. Config
        # key humans_settle_final_round - the arm/control split of the
        # hypothesis test; the control worlds keep the old timing.
        if self.p.get('humans_settle_final_round') \
                and getattr(self, 'initial_trade_plan', None) \
                and (self.model.schedule.time % 1) < 0.9:
            return
        for buynum, buy in enumerate(self.message['trades']):
            # '_nomoney': her fixed money's refusal is final for the board, like any refused draw (the
            # same seller at the same price can only be refused again: money never grows within a board)
            if buy['type'] == 'buy' and 'offers' in buy and buy['offers'] and 'chosen' not in buy \
                    and '_nomoney' not in buy and '_refused' not in buy:
                # Gated demand is banned (the curator 2026-07-20): all demand is
                # present from birth; hard wants go unfilled until the
                # society can actually make them, and that is the frontier.
                weighted_choices = OrderedDict([(offernum, offer['probability']) \
                            for offernum, offer in enumerate(buy['offers'])])
                # Individual recognition is banned from this market
                # (the curator 2026-07-20): signs are the only allocative
                # channel. Audit order is sign-similarity alone.
                sorted_choices = sorted(weighted_choices.items(), key=lambda x:x[1],reverse=True)
                # THE market law (2026-07-14, the curator: "make it to their mutual
                # advantage if a single seller appears"): the walk is still
                # sign-ordered (signs keep their allocative job — first mover
                # among the satisfying), but there is no static bar. The buyer
                # stops at the first offer that beats its ASPIRATION — an
                # exponential moving average of the scores it has been
                # settling at — and only walks deeper when the familiar
                # sellers stop impressing. If nothing beats the habit, the
                # best tested offer settles anyway (never-veto preserved, and
                # the aspiration then decays toward what the market can
                # actually deliver). A lone better seller is therefore FOUND
                # (dissatisfaction deepens the walk), PAID A PREMIUM (the
                # score-scaled price below), and REWARDED IN THE BUYER'S
                # FITNESS (the aspiration-relative notification) — mutual
                # advantage at the single-pioneer event, which is the state
                # the fixed-bar market could never reach. (That old law:
                # git history only, per the no-flags ruling.)
                if len(sorted_choices) > 1:
                    # Restore the original "roll to pick" semantics (the
                    # comment above predates the sorted walk of c0612cd):
                    # the testing ORDER is sampled by sign-similarity
                    # weight, epsilon-floored so even the worst-ranked
                    # seller is occasionally examined first. Every tested
                    # offer's seller is paid its score signal
                    # (seller_score_notification in pass_all_tests), so
                    # over periods every challenger gets evaluations —
                    # the feedback that the deterministic walk starved.
                    eps = 0.02
                    pool = [(num, prob + eps) for num, prob in sorted_choices]
                    rolled = []
                    while pool:
                        tot = sum(w for _, w in pool)
                        r = random.uniform(0, tot)
                        acc = 0.0
                        for i, (num, w) in enumerate(pool):
                            acc += w
                            if r <= acc:
                                rolled.append((num, w))
                                pool.pop(i)
                                break
                        else:
                            rolled.append(pool.pop())
                    sorted_choices = rolled
                found = False
                best = None   # (max_score, winning_num, gepResult, pickle_name)
                count = 0
                # One-draw chooser (the curator, 2026-08-06, config arm): when
                # chooser_draws is set, the buyer examines only that many
                # weighted draws, one in her test, and a draw that does
                # not meet the aspiration is refused, no trade this step.
                _draws = self.p.get('chooser_draws')
                _bok = self.p.get('chooser_best_of_k')
                _sf = self.p.get('sign_first_matching')
                _closest = self.p.get('chooser_closest_only')
                if _closest:
                    # HER DESIGN (2026-08-17, "that combined with
                    # aspiration should work"): the buyer examines THE
                    # sign-closest offer only, deterministically, and
                    # the per-item aspiration gates the settle — a
                    # refused judgment is no trade this step, with the
                    # two-way aspiration breathing of her Aug 6 draws
                    # rule. Implemented as chooser_draws=1 over a
                    # deterministic cosine-descending order so every
                    # aspiration semantic below applies unchanged.
                    # This is the maximal-gain pole of the sign dial
                    # (BLIND 0, raw-cosine whisper, STRONG floor,
                    # CLOSEST everything).
                    _draws = 1
                    _bok = None
                    sorted_choices = sorted(
                        [(n9, float(buy['offers'][n9].get('cosine_sim',
                                                          0.0)))
                         for n9 in range(len(buy['offers']))],
                        key=lambda t9: -t9[1])
                if _sf:
                    # HER RULE (2026-08-11): the offers ARE the funnel
                    # survivors (<= n). With live tests: judge ALL
                    # survivors, best verified score settles. With no
                    # tests: the CLOSEST SIGN settles untested at full
                    # price (deterministic - not the old first-draw).
                    _live = False
                    for _td7 in buy.get('tests', []):
                        if _td7.get('stophere'):
                            break
                        if stop_cut(str(_td7.get('test', ''))) \
                                and stop_cut(str(_td7.get('data', ''))):
                            _live = True
                            break
                    if _live:
                        for _n5 in range(len(buy['offers'])):
                            _pa5, _gr5, _ms5, _pn5 = self.pass_all_tests(buy, _n5)
                            if _pa5 and (best is None or _ms5 > best[0]):
                                best = (_ms5, _n5, _gr5, _pn5)
                    elif buy['offers']:
                        _n6 = max(range(len(buy['offers'])),
                                  key=lambda _k6: buy['offers'][_k6].get('cosine_sim', 0.0))
                        _pa6, _gr6, _ms6, _pn6 = self.pass_all_tests(buy, _n6)
                        if _pa6:
                            best = (_ms6, _n6, _gr6, _pn6)
                    if best is not None:
                        found = True
                        _found_at = 1786
                        max_score, winning_num, gepResult, pickle_name = best
                elif self.p.get('chooser_roulette'):
                    # ONE WEIGHTED DRAW PER BUY ROW PER BOARD (her design
                    # 2026-09-30, rulings 1-2 Oct): pick by sign alone.
                    # Within the category-gated pool, weight = (closeness -
                    # pool minimum)^ticket_power (her lottery law); ONE
                    # draw, remembered on the buy row for the whole board;
                    # no threshold, no walk, no second draw. If the drawn
                    # seller's own parts are not settled yet the buyer
                    # waits (the arranging rounds are for that); when ready
                    # the good is auditioned and TAKEN; junk scores zero
                    # and pays zero at settlement. The buyer that pointed
                    # badly eats nothing this board - that is the pressure
                    # on the sought sign; the seller nobody points at is
                    # never drawn - that is the pressure on the display.
                    _n9 = buy.get('_draw')
                    if _n9 is None and buy['offers']:
                        _cs9 = [float(o9.get('cosine_sim', 0.0))
                                for o9 in buy['offers']]
                        _mn9 = min(_cs9)
                        _e9 = float(self.p.get('ticket_power', 1) or 1)
                        _ws9 = [max(c9 - _mn9, 0.0) ** _e9 for c9 in _cs9]
                        if self.p.get('chooser_blind'):
                            # THE SIGN-BLIND CONTROL (her August signs experiment) in the one-draw design. Found 6 Oct
                            # 00:15: chooser_blind was read only by the older chooser, so since the one-draw design of
                            # 30 Sep a "blind" world drew by sign exactly like its main world (identical counts on the
                            # first read). Every offer now weighs the same; prices, types and tests are untouched.
                            _ws9 = [1.0 for _c9 in _cs9]
                        _tot9 = sum(_ws9)
                        if _tot9 <= 0.0:
                            _n9 = random.randrange(len(_cs9))
                        else:
                            _rr9 = random.uniform(0.0, _tot9)
                            _n9 = len(_cs9) - 1
                            for _j9, _w9 in enumerate(_ws9):
                                _rr9 -= _w9
                                if _rr9 <= 0.0:
                                    _n9 = _j9
                                    break
                        buy['_draw'] = _n9
                    if self.p.get('gates_after_sign') and _n9 is not None and _n9 < len(buy['offers']):
                        # HER RULING (5 Oct 2026, point 1): the drawn offer is checked after the draw;
                        # the wrong type or a price outside the bands is no trade this board.
                        _og9 = buy['offers'][_n9]
                        if not _og9.get('_cat_ok', True):
                            buy['_refused'] = 'type'
                            continue
                        if not _og9.get('_price_ok', True):
                            buy['_refused'] = 'price'
                            continue
                    if _n9 is not None and _n9 < len(buy['offers']):
                        _off9 = buy['offers'][_n9]
                        _ready9 = True
                        if self.p.get('require_complete_inputs'):
                            try:
                                _sid9 = int(_off9['agent'])
                                _s09 = int(_off9['trades'][0])
                                _wtr9 = self.b[_sid9].get('trades', [])
                                for _j9 in range(_s09 + 1, len(_wtr9)):
                                    _t9 = _wtr9[_j9]
                                    if (_t9.get('type') in ('sell', 'stop')
                                            or str(_t9.get('item', '')
                                                   ).startswith('stop')):
                                        break
                                    if (_t9.get('type') == 'buy'
                                            and _t9.get('chosen') is None):
                                        _ready9 = False
                                        break
                            except (TypeError, ValueError, KeyError, IndexError):
                                _ready9 = True
                        if self.p.get('pass_diag'):
                            self._pass_diag('roulette-%s' % ('ready' if _ready9 else 'wait'))
                        if _ready9:
                            _pa9, _gr9, _ms9, _pn9 = \
                                self.pass_all_tests(buy, _n9)
                            if _gr9 is None:
                                try:
                                    _gr9 = self.modular_gep(
                                        self.model.emergent_functions[
                                            self.obtain_trade_list(_off9)])
                                except Exception:
                                    _gr9 = None
                            best = (_ms9, _n9, _gr9, _pn9)
                            found = True
                            _found_at = 1854
                            max_score, winning_num, gepResult, \
                                pickle_name = best
                elif self.p.get('chooser_runnable_closest'):
                    # HER FINAL LAW (2026-08-19, this evening's ruling):
                    #   prefilter 1: price bands overlap;
                    #   prefilter 2: the program RUNS (the chain
                    #     materializes - the census's 43.5% 'noitem'
                    #     candidates never compete);
                    #   among those, THE CLOSEST SIGN SETTLES, always;
                    #   pay according to score - and (her ruling on the
                    #   fork): if a test exists and cannot run on the
                    #   delivered item, or scores nothing, the seller
                    #   is PAID ZERO TOKENS and a human buyer books
                    #   ZERO utility fulfilment (applied at the settle
                    #   block below). Test-less buys keep the standing
                    #   null law. Family membership is NEVER filtered -
                    #   the sign must carry it, and the pay teaches it.
                    # runnable_closest_blind (control): uniform draw
                    # from the same runnable pool.
                    _rpool = []
                    for _n8 in range(len(buy['offers'])):
                        try:
                            _off_c = buy['offers'][_n8]
                            _sid_c = int(_off_c.get('agent'))
                            # F6 fix (2026-09-11): the band check reads THIS offer's
                            # own sell row, not the seller's first sell row.
                            _sell_c = self.b[_sid_c]['trades'][_off_c['trades'][0]]
                        except (TypeError, ValueError, KeyError):
                            continue
                        if (_sell_c is not None and 'midpoint' in buy
                                and 'midpoint' in _sell_c):
                            _bl = max(buy['midpoint']
                                      - buy.get('range', 0.0), 0.0)
                            _bh = buy['midpoint'] + buy.get('range', 0.0)
                            _sl = max(_sell_c['midpoint']
                                      - _sell_c.get('range', 0.0), 0.0)
                            _sh = _sell_c['midpoint'] \
                                + _sell_c.get('range', 0.0)
                            if min(_bh, _sh) < max(_bl, _sl):
                                continue
                        try:
                            _fn8 = self.obtain_trade_list(
                                buy['offers'][_n8])
                            _il8 = self.model.emergent_functions.get(
                                _fn8)
                        except Exception:
                            _il8 = None
                        if _il8:
                            # HER RULING (m1, 2026-08-21): A SELLER WHOSE OWN
                            # SECTION STILL HAS AN UNSETTLED BUY IS NOT READY TO
                            # SELL. Until now the unfilled element was simply
                            # dropped at assembly - the `only_existing` filter
                            # removes a bare family name that resolves to no
                            # registry key - and the SHORTER chain still ran, so
                            # nothing stopped the sale. Measured: 117 of 124
                            # composing sellers shipped missing a part. The
                            # candidate now waits until it has finished its own
                            # shopping; the arranging rounds exist for exactly
                            # that.
                            _ready8 = True
                            if self.p.get('require_complete_inputs'):
                              try:
                                  _wtr8 = self.b[_sid_c].get('trades', [])
                                  _s08 = int(_off_c.get('trades', [0])[0])
                                  for _j8 in range(_s08 + 1, len(_wtr8)):
                                      _t8 = _wtr8[_j8]
                                      if (_t8.get('type') in ('sell', 'stop')
                                              or str(_t8.get('item', '')
                                                     ).startswith('stop')):
                                          break
                                      if (_t8.get('type') == 'buy'
                                              and _t8.get('chosen') is None):
                                          _ready8 = False
                                          break
                              except (TypeError, ValueError, KeyError, IndexError):
                                  _ready8 = True
                            if _ready8:
                                _rpool.append(_n8)
                    buy['_rpool'] = list(_rpool)
                    if self.p.get('chooser_sign_walk'):
                        # THE WALK: candidates refused earlier on THIS
                        # board are not offered again. The list lives on
                        # the buy, so it clears when the board rebuilds
                        # the plan - a fresh board is a fresh search.
                        _seen8 = buy.get('_walked') or []
                        _rem8 = [_k8 for _k8 in _rpool
                                 if _k8 not in _seen8]
                        if _rem8:
                            _rpool = _rem8
                    if _rpool:
                        if self.p.get('runnable_closest_blind'):
                            _nw8 = _rpool[
                                random.randrange(len(_rpool))]
                        else:
                            _nw8 = max(_rpool, key=lambda _k8: float(
                                buy['offers'][_k8].get(
                                    'cosine_sim', 0.0)))
                        buy['_rwin'] = _nw8
                        _pa8, _gr8, _ms8, _pn8 = \
                            self.pass_all_tests(buy, _nw8)
                        # THE THRESHOLD LAW (her revival, 2026-08-20):
                        # in test_threshold worlds a buy with live
                        # tests settles ONLY when the score EXCEEDS its
                        # evolved bar; a refusal is no trade this round
                        # and the buy retries - failure returns, the
                        # twenty rounds refill, for humans and learners
                        # alike. Replaces humans-settle-last. Test-less
                        # buys keep the null lane.
                        _live8 = False
                        _bar8 = 0.0
                        if self.p.get('test_threshold'):
                            for _td8 in buy.get('tests', []):
                                if _td8.get('stophere'):
                                    break
                                if stop_cut(str(_td8.get('test', ''))) \
                                        and stop_cut(str(_td8.get(
                                            'data', ''))):
                                    _live8 = True
                                    _bar8 = float(_td8.get(
                                        'threshold', 0.0))
                                    if (self.p.get(
                                            'unrunnable_test_is_no_test')
                                            and not self
                                            .test_names_something_real(
                                                _td8.get('test'))):
                                        # names nothing real - untested lane
                                        _live8 = False
                                        _bar8 = 0.0
                                break
                        # HER SPEC (2026-08-20): a bar of ZERO means
                        # NOTHING - accept anything, like settle-always
                        # (zero-pay if unscoreable). Refusal exists
                        # only when the bar has been RAISED above zero.
                        _tok8 = bool(buy.get('_type_ok', 1))
                        if self.p.get('chooser_sign_walk') and (
                                (_live8 and not _tok8)
                                or (_live8 and _bar8 > 0.0 and not _pa8)):
                            # remember this refusal for the rest of the
                            # board, so the next round walks on
                            _w8 = buy.get('_walked')
                            if _w8 is None:
                                _w8 = []
                                buy['_walked'] = _w8
                            if _nw8 not in _w8:
                                _w8.append(_nw8)
                        if _live8 and not _tok8:
                            # refused: the WRONG KIND of good - the
                            # want's own test could not run on it.
                            # Binds at EVERY bar, zero included.
                            pass
                        elif _live8 and _bar8 > 0.0 and not _pa8:
                            pass  # refused: below the bar, retry later
                        else:
                            if _gr8 is None:
                                # product needed even when no test could
                                # touch it (zero-pay junk, non-threshold
                                # worlds)
                                try:
                                    _gr8 = self.modular_gep(
                                        self.model.emergent_functions[
                                            self.obtain_trade_list(
                                                buy['offers'][_nw8])])
                                except Exception:
                                    _gr8 = None
                            best = (_ms8, _nw8, _gr8, _pn8)
                            found = True
                            _found_at = 2020
                            max_score, winning_num, gepResult, \
                                pickle_name = best
                elif self.p.get('chooser_closest_tested'):
                    # HER CORRECTION (2026-08-19, on seeing the
                    # A-family's 100% settle rate: "sounds like a bug.
                    # if you point in a random place, and you have a
                    # pinned test that only works with a clusterer,
                    # then the closest sign match might not be a
                    # clusterer... in that case you would get
                    # nothing."): the audit-first reading guaranteed
                    # every aim a working seller and erased the aim's
                    # token consequence. THE LAW AS MEANT:
                    #   1. pool = the PRICE-OVERLAPPING offers only
                    #      (her "has overlap in price");
                    #   2. THE CLOSEST SIGN in that pool is examined,
                    #      ALONE;
                    #   3. it settles only if its program actually
                    #      works (pass_all_tests: runs AND passes; the
                    #      test-less bootstrap lane settles under the
                    #      standing null law);
                    #   4. if it fails, NOTHING settles this step - no
                    #      walk, no second-closest. Aim badly, earn
                    #      nothing: the sought sign is under selection
                    #      at last.
                    # closest_tested_blind (control): uniform draw
                    # from the same band pool, same test-or-nothing.
                    # Pool, winner, and failure are logged on the buy
                    # (_bpool/_bwin/_bfail) for exact gate checks.
                    _bpool = []
                    for _n8 in range(len(buy['offers'])):
                        try:
                            _off_c = buy['offers'][_n8]
                            _sid_c = int(_off_c.get('agent'))
                            _sell_c = next(
                                (t for t in self.b[_sid_c].get(
                                    'trades', [])
                                 if t.get('type') == 'sell'), None)
                        except (TypeError, ValueError, KeyError):
                            _sell_c = None
                        if (_sell_c is None or 'midpoint' not in buy
                                or 'midpoint' not in _sell_c):
                            # Bands unreadable here: keep the offer;
                            # pass_all_tests' own band law is the
                            # authoritative refusal.
                            _bpool.append(_n8)
                            continue
                        _bl = max(buy['midpoint']
                                  - buy.get('range', 0.0), 0.0)
                        _bh = buy['midpoint'] + buy.get('range', 0.0)
                        _sl = max(_sell_c['midpoint']
                                  - _sell_c.get('range', 0.0), 0.0)
                        _sh = _sell_c['midpoint'] \
                            + _sell_c.get('range', 0.0)
                        if min(_bh, _sh) >= max(_bl, _sl):
                            _bpool.append(_n8)
                    buy['_bpool'] = list(_bpool)
                    if _bpool:
                        if self.p.get('closest_tested_blind'):
                            _nw8 = _bpool[
                                random.randrange(len(_bpool))]
                        else:
                            _nw8 = max(_bpool, key=lambda _k8: float(
                                buy['offers'][_k8].get(
                                    'cosine_sim', 0.0)))
                        buy['_bwin'] = _nw8
                        _pa8, _gr8, _ms8, _pn8 = \
                            self.pass_all_tests(buy, _nw8)
                        if _pa8:
                            best = (_ms8, _nw8, _gr8, _pn8)
                            found = True
                            _found_at = 2090
                            max_score, winning_num, gepResult, \
                                pickle_name = best
                        else:
                            buy['_bfail'] = True
                elif self.p.get('chooser_best_eligible'):
                    # HER LAW (2026-08-19, the mimicry-pressure market;
                    # her design: "take only the best sign that has a
                    # working program and has overlap in price, and pay
                    # it in accord with its score"). Three moves, all
                    # composed from standing law:
                    #   1. ELIGIBILITY FIRST - a candidate competes
                    #      only if its price band overlaps the buy (the
                    #      Aug-18 band law, refused inside
                    #      pass_all_tests before any program runs) AND
                    #      its offered program actually RUNS (a raising
                    #      audition produces no result). Wants with no
                    #      live tests audition nothing; there
                    #      eligibility is the band gate alone, so the
                    #      token-only bootstrap lane stays open.
                    #   2. THE CLOSEST SIGN AMONG ELIGIBLES SETTLES,
                    #      deterministically - no lottery, no k, no
                    #      aspiration refusal. The sign wins the
                    #      BUSINESS; ticket_power and k are inert in
                    #      this family.
                    #   3. THE SCORE SETS THE PAY - the quality-market
                    #      law downstream, untouched.
                    # A junior wearing a near-copy of a rich badge owns
                    # every buyer whose pointing lands nearer its
                    # variant - the foothold the powered lottery
                    # diluted away. Every audited candidate also gets
                    # its score signal, so losers receive the feedback
                    # to improve: the selective pressure she named.
                    # best_eligible_blind (the control): a uniform draw
                    # among the SAME eligibles - one element between
                    # the arms, the choice rule itself.
                    _live8 = False
                    for _td8 in buy.get('tests', []):
                        if _td8.get('stophere'):
                            break
                        if stop_cut(str(_td8.get('test', ''))) \
                                and stop_cut(str(_td8.get('data', ''))):
                            _live8 = True
                            break
                    _elig8 = []
                    for _n8 in range(len(buy['offers'])):
                        _pa8, _gr8, _ms8, _pn8 = \
                            self.pass_all_tests(buy, _n8)
                        _ok8 = (_gr8 is not None) if _live8 \
                            else bool(_pa8)
                        if _ok8:
                            _elig8.append((_n8, _gr8, _ms8, _pn8))
                    # The eligibility set is logged with the trade so
                    # the launch gate can verify the law exactly.
                    buy['_elig'] = [_t8[0] for _t8 in _elig8]
                    if _elig8:
                        if self.p.get('best_eligible_blind'):
                            _n8, _gr8, _ms8, _pn8 = _elig8[
                                random.randrange(len(_elig8))]
                        else:
                            _n8, _gr8, _ms8, _pn8 = max(
                                _elig8, key=lambda _t8: float(
                                    buy['offers'][_t8[0]].get(
                                        'cosine_sim', 0.0)))
                        best = (_ms8, _n8, _gr8, _pn8)
                        found = True
                        _found_at = 2155
                        max_score, winning_num, gepResult, \
                            pickle_name = best
                elif _bok or self.p.get('chooser_closest_always') \
                        or self.p.get('chooser_closest_k'):
                    # Best-of-k chooser (the curator, 2026-08-07): candidates
                    # weighted by the RAW two-sided cosine similarity of
                    # April 4, the buyers_weight blend of buyer-seeks-
                    # seller and seller-seeks-buyer, no floor subtracted.
                    # k distinct candidates drawn at random by weight,
                    # ALL k judged, and the best verified score among
                    # those that pass settles. No aspiration refusal.
                    #
                    # chooser_closest_always (HER DESIGN 2026-08-17,
                    # "take it every time, just paid in comparison to
                    # how well the last guy did"): the drawn set is
                    # exactly THE closest sign, judged, and it settles
                    # unconditionally when it runs — quality decides
                    # the PRICE (score over best-ever frontier,
                    # downstream), never the purchase. The gate
                    # pathologies (ratchet freeze, stuck-low) are
                    # impossible by construction.
                    # BLIND/STRONG live HERE now (2026-08-17): the
                    # 2026-08-15 switches sat at the walk-path weights
                    # (~line 350), which the k-family examination never
                    # reads — both signs arms bit-replayed the reference
                    # world (md5-identical metrics, caught by her
                    # side-by-side demand). chooser_blind: every offer
                    # weighs 1 (uniform draws, the pure-Hayek arm).
                    # sign_floor_subtract F: weight = max(cos - F, 0),
                    # the amplified arm. Default: raw cosine.
                    _blind9 = bool(self.p.get('chooser_blind'))
                    _floor9 = self.p.get('sign_floor_subtract')
                    _cos9 = [float(buy['offers'][n2].get('cosine_sim', 0.0))
                             for n2 in range(len(buy['offers']))]
                    # HER LOTTERY LAW (2026-08-18, correcting the Aug-7
                    # raw-cosine deviation): NORMALIZE by subtracting the
                    # pool's LOWEST similarity — the same minus-the-field-
                    # minimum law as the walk path and the judgment
                    # lottery. The worst-matched candidate earns zero
                    # tickets; everyone else competes on excess over it.
                    # When all similarities are equal the weights all hit
                    # zero and the draw falls back to uniform — equal
                    # signs carry no information, so that is correct.
                    # sign_floor_subtract (the arbitrary-constant form)
                    # is superseded and kept only for record replay.
                    _mn9 = min(_cos9) if _cos9 else 0.0
                    _pw9 = float(self.p.get('ticket_power', 1) or 1)

                    def _w9(_c9):
                        if _blind9:
                            return 1.0
                        if _floor9 is not None:
                            return max(_c9 - float(_floor9), 0.0)
                        # HER GRADIENT LEVER 2 (2026-08-17): ticket
                        # power sharpens the lawful wheel - ratios
                        # square at power 2; near-ties stay near-tied.
                        return max(_c9 - _mn9, 0.0) ** _pw9
                    _pool = [(n2, _w9(_cos9[n2]))
                             for n2 in range(len(_cos9))]
                    _drawn = []
                    if self.p.get('chooser_closest_always') and _pool:
                        _drawn = [max(_pool, key=lambda t2: t2[1])[0]]
                        _pool = []
                    _ck = self.p.get('chooser_closest_k')
                    if _ck and _pool:
                        # HER DESIGN (2026-08-17): examine the closest
                        # k signs DETERMINISTICALLY (no roulette), judge
                        # all of them, best verified score settles — the
                        # determinism ladder of the sign dial (closest-5,
                        # closest-2 vs the weighted-draw k of the rungs).
                        _pool.sort(key=lambda t2: -t2[1])
                        _drawn = [n2 for n2, _w2 in _pool[:int(_ck)]]
                        _pool = []
                    for _ in range(min(_bok or 0, len(_pool))):
                        _tot = sum(w for _, w in _pool)
                        if _tot <= 0:
                            _idx = random.randrange(len(_pool))
                        else:
                            _rr = random.uniform(0.0, _tot)
                            _idx = len(_pool) - 1
                            for _j2, (_n3, _w3) in enumerate(_pool):
                                _rr -= _w3
                                if _rr <= 0:
                                    _idx = _j2
                                    break
                        _drawn.append(_pool.pop(_idx)[0])
                    _cred = float(self.p.get('credulous_settle_prob',
                                              0.0) or 0.0)
                    if _drawn and _cred > 0 and random.random() < _cred:
                        # HER RULING (2026-08-18): a want ALWAYS says
                        # what it wants — the ask carries its test.
                        # Corruption is the buyer SETTLING WITHOUT
                        # RUNNING it: the credulous buyer takes the
                        # lottery's first pick untested (score-0
                        # settle: option-A price, experience-goods
                        # silent scoring, audit pool — all existing
                        # untested-settle law applies).
                        _nc = _drawn[0]
                        _sh = dict(buy)
                        _sh['tests'] = [{'stophere': True}]
                        _pa, _gr, _ms, _pn = self.pass_all_tests(_sh, _nc)
                        if _pa:
                            best = (0.0, _nc, _gr, _pn)
                            found = True
                            _found_at = 2259
                            max_score, winning_num, gepResult,                                 pickle_name = best
                    if not found:
                        # (the per-buy fork warmer that lived here
                        # failed its gate 10x slower and was removed;
                        # the board-level sharded warm in MarketSim.step
                        # is the cores mechanism now)
                        for _n4 in _drawn:
                            _pa, _gr, _ms, _pn = self.pass_all_tests(buy, _n4)
                            if _pa and (best is None or _ms > best[0]):
                                best = (_ms, _n4, _gr, _pn)
                        if best is not None:
                            found = True
                            _found_at = 2271
                            max_score, winning_num, gepResult,                                 pickle_name = best
                def _cap_legacy():
                    return 0 if (_bok or self.p.get('chooser_closest_always')
                                 or self.p.get('chooser_closest_k')
                                 # best-eligible law: no eligibles means NO
                                 # TRADE - the walk fallback must not hand
                                 # the buy to an ineligible candidate.
                                 or self.p.get('chooser_best_eligible')
                                 # closest-tested law: a failed closest is
                                 # NO TRADE - the walk must never run.
                                 or self.p.get('chooser_closest_tested')
                                 # runnable-closest law: the branch settles
                                 # or the pool was empty - never the walk.
                                 or self.p.get('chooser_runnable_closest')) else (
                        min(_draws, len(sorted_choices)) if _draws
                        else len(sorted_choices))
                # Under chooser_roulette the April-4 ordered loop below must never
                # run: it is the walk (audition the candidates in sign order until
                # one passes), which her 1 Oct ruling removed. Found 3 Oct 2026:
                # with roulette missing from this list, every draw that was not
                # ready fell through to the walk - 9 of 29 smoke settlements,
                # auditioned without the readiness wait.
                _cap = 0 if self.p.get('chooser_roulette') else _cap_legacy()
                while not found and count < _cap:
                    winning_num = sorted_choices[count][0]
                    count+=1
                    pass_all_tests, gepResult, max_score, pickle_name = self.pass_all_tests(buy, winning_num)
                    if pass_all_tests:
                        if best is None or max_score > best[0]:
                            best = (max_score, winning_num, gepResult, pickle_name)
                        # Aspiration keys per (agent, item) exactly like the
                        # frontier — one habit per need, not one blended
                        # habit per agent (bug caught 2026-07-15).
                        _akey = stop_cut(buy['item'])
                        if max_score < getattr(self, '_aspiration_by', {}).get(_akey, 0.0):
                            continue   # keep walking: habit not beaten
                        best = (max_score, winning_num, gepResult, pickle_name)
                        found = True
                        _found_at = 2301
                if not found and best is not None and _draws and not _bok:
                    # Two-way aspiration (the curator, 2026-08-06: tiny
                    # downwards): a refused judgment lowers the bar
                    # slightly toward the disappointing score, so
                    # standards breathe and the one-way ratchet cannot
                    # freeze a want. Default down-rate one tenth of
                    # the 0.05 up-rate.
                    _dk = stop_cut(buy['item'])
                    if not hasattr(self, '_aspiration_by'):
                        self._aspiration_by = {}
                    _a = self._aspiration_by.get(_dk, 0.0)
                    if best[0] < _a:
                        _down = self.p.get('aspiration_down', 0.005)
                        self._aspiration_by[_dk] = max(
                            0.0, _a - _down * (_a - max(0.0, best[0])))
                if not found and best is not None and not _draws and not _bok:
                    # Never-veto fallback: settle the best seen below
                    # aspiration. Skipped under chooser_draws, her
                    # one-draw variant refuses below-aspiration draws.
                    found = True
                # Her null law, final form (2026-08-09, repealing the
                # Aug 5 bootstrap/strict-null gates): a stop codon in
                # the test place settles the buy at the full negotiated
                # price without testing — for any buyer, always — and
                # learning, not law, teaches agents to carry tests.
                # A zero score is still a missing measurement, not a
                # low one, so the quality law never scales it.
                if found and getattr(self, '_money_left', None) is not None:
                    # HER FIXED MONEY (4 Oct 2026): rows are paid in their order from the board's money; a
                    # purchase that would cost more than what is left does not happen - no sale, the seller
                    # is paid nothing and the human gets nothing, as with a refused band. The cost is
                    # computed by _would_pay exactly as the settle below computes it, before any record.
                    _owe = self._would_pay(buy, best)
                    if _owe > self._money_left + 1e-9:
                        found = False
                        buy['_nomoney'] = round(_owe, 6)
                        self._money_row(buy, _owe, 'no money')
                if found:
                    if self.p.get('pass_diag'):
                        self._pass_diag('settled-by;%s' % locals().get('_found_at', '?'))
                    max_score, winning_num, gepResult, pickle_name = best
                    buy['chosen'] = winning_num
                    sell = self.b[buy['offers'][winning_num]['agent']]['trades'][buy['offers'][winning_num]['trades'][0]]
                    buy['price'] = self.price(buy, sell)
                    _pl = self.p.get('price_list')
                    if _pl:
                        # THE PRICE LIST (her ruling 2026-10-01: price set
                        # aside; how it returns is decided at the time). A
                        # sale pays the authored price of the want if the
                        # buyer authored one, else the family's list price;
                        # no band, no bargaining; the learned price floats
                        # stay in the genome unread.
                        _fam4 = stop_cut(str(sell.get('item', ''))).split('_')[0]
                        _price4 = _pl.get(_fam4, _pl.get('default', buy['price']))
                        for _mt4 in (getattr(self, 'initial_trade_plan', None)
                                     or {}).get('trades', []):
                            if (_mt4.get('type') == 'buy'
                                    and stop_cut(str(_mt4.get('item', ''))).split('_')[0] == _fam4
                                    and _mt4.get('midpoint') is not None):
                                _price4 = float(_mt4['midpoint'])
                                break
                        if buy.get('cd_good') and self.p.get('humans_cd'):
                            _price4 = float(self.p['humans_cd']['prices'].get(buy['cd_good'], _price4))
                        buy['price'] = float(_price4)
                    key = (self.unique_id, stop_cut(buy['item']))
                    if not hasattr(self.model, '_quality_frontier'):
                        self.model._quality_frontier = {}
                    _prev_fr = self.model._quality_frontier.get(key, 0.0)
                    frontier = max(_prev_fr, max_score)
                    self.model._quality_frontier[key] = frontier
                    # Convergence-novelty marks (the curator, 2026-08-05): a
                    # frontier record or a never-before-traded item
                    # resets the convergence stop's countdown in go().
                    self.model._settle_count = getattr(
                        self.model, '_settle_count', 0) + 1
                    if not hasattr(self.model, '_items_seen'):
                        self.model._items_seen = set()
                    _nov = frontier > _prev_fr
                    if buy['item'] not in self.model._items_seen:
                        self.model._items_seen.add(buy['item'])
                        _nov = True
                    if _nov:
                        self.model._last_novelty_iter = getattr(
                            self.model, '_iter', 0)
                    _dbg_mid = buy['price']
                    if self.p.get('chooser_runnable_closest') or self.p.get('chooser_roulette'):
                        # HER ZERO-PAY RULING (2026-08-19): "if there
                        # is a test, and the test cant be run or
                        # returns a bad score, the offerer gets paid
                        # zero tokens, and the human gets zero
                        # fulfilment of utility." The junk trade still
                        # settles - the seller wastes its slot for
                        # nothing, so every board grades every seller.
                        # Test-less buys keep the null law's full
                        # price. (max_score covers both cases: a test
                        # that never ran and a test that scored 0.)
                        _zp_live = False
                        for _td0 in buy.get('tests', []):
                            if _td0.get('stophere'):
                                break
                            if stop_cut(str(_td0.get('test', ''))) \
                                    and stop_cut(str(_td0.get(
                                        'data', ''))):
                                _zp_live = True
                            break
                        if _zp_live and max_score <= 0 and self.p.get('pay_by_score', True):
                            buy['price'] = 0.0
                            buy['_zeropay'] = 1
                    # A zero score here always means no test was runnable
                    # (a measured zero cannot settle), so unmeasured goods
                    # pay the negotiated price for every buyer (the curator's
                    # option A, 2026-08-05); only measured quality is
                    # scaled by her law.
                    if max_score > 0 and frontier > 0:
                        # THE QUALITY LAW (her Jul 14 design, kept whole):
                        # payment scaled by score over the FRONTIER, the
                        # best score this buyer has seen for this want.
                        # The frontier is the pioneer premium: whoever
                        # raises it is paid in full, everyone below it
                        # is discounted - that is what makes search
                        # effort pay. Her 11 Sep addition: make the
                        # differences more pronounced, still priced
                        # with the frontier:
                        #   factor = (score/frontier)^k * frontier
                        #          = score^k / frontier^(k-1)
                        # k = quality_exponent. k=1 is the Jul 14 law
                        # exactly. At k=2 a delivery at the frontier
                        # earns the frontier's own level (a 0.98 clusterer
                        # 0.98, a 0.50 classifier 0.50), and one below
                        # it is discounted by the square of its
                        # shortfall. (My first form on 11 Sep dropped the
                        # frontier without reading why it was there -
                        # her correction; that form ran only in v8,
                        # killed at once.)
                        if self.p.get('pay_by_score', True):
                            buy['price'] *= self.quality_factor(max_score, frontier)
                        # pay_by_score false (her request 3 Oct 2026: "try paying by learnt
                        # price only, unrelated to score"): the seller is paid the price
                        # negotiated from the two learned bands whatever the score; the
                        # score still sets the buyer's utility, so quality is rewarded only
                        # through the buyer's choosing (being preferred).
                    if getattr(self, '_money_left', None) is not None:
                        # her fixed money: the debit. _would_pay must agree with the price just set;
                        # if it ever does not, the run stops rather than let money go negative unseen.
                        if abs(buy['price'] - _owe) > 1e-6:
                            raise RuntimeError('fixed money: would-pay %.6f but settle price %.6f'
                                               % (_owe, buy['price']))
                        self._money_left -= buy['price']
                        self._money_row(buy, buy['price'], 'paid')
                    _dbgp = os.environ.get('CORE_SETTLE_DEBUG')
                    if _dbgp:
                        # Temporary settlement instrumentation (same as
                        # the main engine's): one row per settlement
                        # with the quantities the ledger never records.
                        try:
                            _new = not os.path.exists(_dbgp)
                            with open(_dbgp, 'a') as _f:
                                if _new:
                                    _f.write('time;buyer;want;score;frontier;'
                                             'factor;midpoint;paid;aspiration\n')
                                _f.write('%s;%s;%s;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f\n' % (
                                    getattr(self.model.schedule, 'time', -1),
                                    self.unique_id, key[1] + ('|' + buy['cd_good'] if buy.get('cd_good') else ''),
                                    max_score, frontier,
                                    self.quality_factor(max_score, frontier)
                                    if (max_score > 0 and frontier > 0) else 1.0,
                                    _dbg_mid, buy['price'],
                                    getattr(self, '_aspiration_by', {}).get(key[1], 0.0)))
                        except Exception:
                            pass
                    _hfa = self.p.get('humans_fitness_authored') \
                        and getattr(self, 'initial_trade_plan', None)
                    if _hfa:
                        # Her v14 law (2026-08-09): humans' fitness is
                        # the ONE authored utility function, computed
                        # at window close in RoleSim.step from the
                        # window's acquisitions; nothing is scored per
                        # settle here beyond recording what was bought
                        # for how much.
                        _acq_score = max_score
                        if (self.p.get('human_tests_learned')
                                and max_score == 0):
                            # Her competing-states law (2026-08-10): an
                            # untested purchase confers its TRUE quality
                            # anyway (experience goods). Score the
                            # delivered chain SILENTLY with this want's
                            # authored test: the utility feels the
                            # truth, the payment stays full price, and
                            # the seller sees nothing.
                            try:
                                _base = stop_cut(buy['item'])
                                _auth = []
                                for _mt in self.initial_trade_plan.get(
                                        'trades', []):
                                    if stop_cut(_mt.get('item', '')) == _base:
                                        _auth = [dict(_t) for _t in
                                                 _mt.get('tests', [])
                                                 if _t.get('test')
                                                 and not _t.get('stophere')]
                                        break
                                if _auth:
                                    _shadow = dict(buy)
                                    _shadow['tests'] = _auth + [
                                        {'stophere': True}]
                                    _sp, _sg, _sq, _sn = \
                                        self.pass_all_tests(_shadow,
                                                            winning_num)
                                    _acq_score = _sq
                                    # The stream write lives in its OWN
                                    # try: a failed WRITE must never void
                                    # the score the utility feels. On the
                                    # boxes the stream dir did not exist,
                                    # and this shared try silently zeroed
                                    # _acq_score for every untested settle
                                    # (caught 2026-08-15; sweeps 321-324
                                    # invalidated; root fix here plus
                                    # mkdir in the launcher).
                                    try:
                                        _tqp = os.environ.get('CORE_TRUEQ')
                                        if _tqp:
                                            with open(_tqp, 'a') as _tf:
                                                _tf.write('%s;%s;%s;%.6f;%.6f;0\n' % (
                                                    getattr(self.model.schedule,
                                                            'time', -1),
                                                    self.unique_id, _base,
                                                    _sq, buy['price']))
                                    except OSError:
                                        pass
                            except Exception:
                                _acq_score = max_score
                        if not hasattr(self, '_window_acq'):
                            self._window_acq = []
                        self._window_acq.append(
                            (stop_cut(buy['item']), _acq_score,
                             buy['price'], buy.get('cd_good')))
                        if max_score == 0 and \
                                self.model.parameters.get('audit_budget', 0):
                            # The audit institution (her scenario,
                            # 2026-08-10): untested settles join the
                            # period's audit pool. Congestion and the
                            # clawback happen in MarketSim at period end.
                            if not hasattr(self.model, '_audit_pool'):
                                self.model._audit_pool = []
                            try:
                                _seller_uid = buy['offers'][winning_num]['agent']
                            except Exception:
                                _seller_uid = None
                            self.model._audit_pool.append(
                                (self, dict(buy), winning_num,
                                 float(buy['price']), _seller_uid,
                                 float(frontier)))
                    if (not _hfa) and buy.get('alpha') is not None:
                        # HER FORM A (the curator 2026-07-20: keep the same
                        # function as before AND the utility): the good
                        # being bought is IMPROVEMENT — the same
                        # satisfaction-delta as ever (score minus the
                        # adapting aspiration, floored at zero) — wrapped
                        # in Cobb-Douglas with budget kept, so the
                        # genome-learned price faces a real trade-off.
                        # Habitual quality still pays nothing (perpetual
                        # improvement pressure intact); improvement pays
                        # more when bought cheaper. Wants are authored;
                        # prices are never authored.
                        _a = float(buy['alpha'])
                        _B = max(1.0, float(buy.get('budget', 1.0)))
                        _m = max(0.0, min(1.0, (_B - buy['price']) / _B))
                        _ik = stop_cut(buy['item'])
                        _asp = getattr(self, '_aspiration_by', {}).get(_ik, 0.0)
                        _d = max(0.0, min(1.0, max_score - _asp))
                        _U = (_d ** _a) * (_m ** (1.0 - _a)) if _d > 0 else 0.0
                        self.buyer_score_notification(_U, buynum)
                    buy['code']= gepResult
                    # pickle_name can be empty when perform_test returned a score
                    # without a pickle reference (compositional trade paths). Guard the lookup.
                    buy['pickle']= self.model.pickles.get(pickle_name) if pickle_name else None
                    # Aspiration-relative fitness: settling at the habitual
                    # level pays ~0; improvement pays the margin. Then the
                    # habit adapts (Simon-style aspiration), per need.
                    if not hasattr(self, '_aspiration_by'):
                        self._aspiration_by = {}
                    item_key = stop_cut(buy['item'])
                    asp = self._aspiration_by.get(item_key, 0.0)
                    if (not _hfa) and buy.get('alpha') is None:
                        # Aspiration-relative fitness for non-utility
                        # buyers only; utility buyers' fitness is the
                        # Cobb-Douglas utility alone (below) — one
                        # objective per agent, never two.
                        self.buyer_score_notification(max_score - asp, buynum)
                    alpha = self.p.get('aspiration_alpha', 0.05)
                    _new_asp = asp + alpha * (max_score - asp)
                    # Anchored standards (her competing-states math,
                    # 2026-08-10): a client's aspiration adapts but
                    # never falls below anchor_frac times the want's
                    # authored quality standard - the clause that makes
                    # "testing in a junk world yields nothing" real,
                    # and with it the vicious basin. The standard rides
                    # in on the mask's config passthrough; wants
                    # without one anchor nowhere (anchor_frac 0 = off).
                    _anchor_frac = self.p.get('aspiration_anchor_frac', 0)
                    _standard = buy.get('standard')
                    if _anchor_frac and _standard:
                        _new_asp = max(_new_asp, _anchor_frac * float(_standard))
                    self._aspiration_by[item_key] = _new_asp
                    if 'distributes' in self.message and  self.message['distributes']:
                        self.distribute_funds(buy, buynum)
                        #self.distribute_funds(buy['offers'][winning_num], buynum)


    # functions that translate the float vec to a trade plan that can go on the blackboard

    def vector_size(self):
        # Genome version 4 (her design, 2026-08-12): THE PLAN SAYS WHAT,
        # AND POINTS AT HOW.
        #
        # "a buy needs to be able to have a test and a sign to seek but a
        #  construct and sell doesnt, so why not point to a block that
        #  includes the test and the sign to seek both."
        #
        # A want - a sign to seek together with the tests that qualify it
        # - is 12 floats, and only a BUY uses one. Carrying a copy in
        # every plan meant every sell and every construct paid twelve
        # floats for nothing. Now a plan carries a one-float POINTER into
        # a small pool of want blocks, which several plans may share: an
        # agent holds a few standing wants instead of inventing a fresh
        # one per plan and relating none of them.
        #
        # Price stays INLINE, and her question was why pointing at it
        # saved nothing. Pointing saves N(B-1) - K*B floats, for N plans,
        # block size B, pool size K. A want has B=12: 6(11) - 3(12) =
        # +30. A price band has B=2: 6(1) - 3(2) = 0 - the pointer costs
        # almost as much as the band, so three blocks is exactly
        # break-even and more is a loss. Pool the big block, carry the
        # small one.
        #
        # Layout: display sign | plans | want blocks | test bank.
        return (
            self.p['sign_size'] +
            self.n_plans() * self.trade_size() +
            self.want_block_count() * self.want_block_size() +
            self.p['test_bank_size'] * self.bank_entry_size())

    def is_cd_human(self):
        """HER COBB-DOUGLAS SCENARIO (4 Oct 2026, "yes unpin and make cd scenario"): a human that
        wants every good. Decided once, from the authored entry, before the genome is laid out."""
        if not hasattr(self, '_cd_human'):
            m = getattr(self, 'message', None)
            self._cd_human = bool(self.p.get('humans_cd')) and isinstance(m, dict) \
                and bool(m.get('cd_human'))
        return self._cd_human

    def n_plans(self):
        """Plans in this agent's genome: num_trade_plans, or for a Cobb-Douglas human its
        rows_per_good rows for every good plus the closing stop."""
        if self.is_cd_human():
            cd = self.p['humans_cd']
            return int(cd.get('rows_per_good', 3)) * len(cd['goods']) + 1
        # AGENTS OF DIFFERENT CAPACITIES (her suggestion, 6 Oct 2026: "differnt abilities to hold information like
        # differnt length genomes"): plans_by_learner, a list cycled by agent number, gives each learner its own
        # number of plan rows. Absent: every learner has num_trade_plans, exactly as before.
        _pl = self.p.get('plans_by_learner')
        if _pl:
            return int(_pl[int(self.unique_id) % len(_pl)])
        return self.p['num_trade_plans']

    def pooled(self):
        """Whether THIS agent's wants are pooled (standing want blocks with a pointer in each buy row) or live inside
        each buy row (a sought sign and four test selectors per row, the original engine's layout). pool_wants off
        turns pooling off for everyone; pool_wants_learners false (7 Oct 2026, her "you should have" on removing
        pooling and pointing) turns it off for learners while a Cobb-Douglas human keeps one standing want per good,
        read through the tree of decision 18, so a human's demand for a good stays one sought sign."""
        if not self.p.get('pool_wants', True):
            return False
        if self.p.get('pool_wants_learners', True) is False and not self.is_cd_human():
            return False
        return True

    def want_block_count(self):
        # Zero when pooling is off: the wants live inside the plans, so
        # there is no shared pool and vector_size drops the term entirely.
        if not self.pooled():
            return 0
        # Three: at most about half of six plans are buys at any moment,
        # so three standing wants is what an agent can actually use.
        if self.is_cd_human():
            return len(self.p['humans_cd']['goods'])
        return int(self.p.get('want_block_count', 3))

    def want_block_size(self):
        # a sign to seek, plus four test selectors (her 2026-08-10
        # sizing: "you should allow 4 tests")
        return self.p['sign_size'] + 4

    def discrete_gene_mask(self):
        """Which genes decode to a CHOICE (a row's type, an item's levels, a want pointer, a Cobb-Douglas human's
        good, the test selectors, the test bank's paths) as against a magnitude (the display sign, a want's sought
        sign, a price band's midpoint and range). Built once from the layout: display sign | plans | want blocks |
        test bank. Used by the dead-zone reading (RoleSim, 7 Oct 2026)."""
        if getattr(self, '_discrete_mask', None) is not None:
            return self._discrete_mask
        S = int(self.p['sign_size'])
        mask = [False] * S
        for _ in range(self.n_plans()):
            row = [True] + [True] * int(self.p['item_size']) + [False, False]
            if not self.pooled():
                row += [False] * S + [True] * 4
            else:
                row += [True] + ([True] if self.is_cd_human() else [])
            assert len(row) == self.trade_size(), (len(row), self.trade_size())
            mask += row
        for _ in range(self.want_block_count()):
            mask += [False] * S + [True] * (self.want_block_size() - S)
        for _ in range(int(self.p['test_bank_size'])):
            mask += [True] * self.bank_entry_size()
        self._discrete_mask = np.array(mask, dtype=bool)
        return self._discrete_mask

    def want_block_offset(self):
        return (self.p['sign_size']
                + self.n_plans() * self.trade_size())

    def want_block(self, afloat):
        """Which standing want a plan points at. The pool is small, so
        the float is cut into equal slices - the same interval reading
        as everything else in the genome."""
        k = self.want_block_count()
        try:
            x = float(afloat)
        except (TypeError, ValueError):
            return 0
        if x < 0.0:
            x = 0.0
        if x >= 1.0:
            x = 1.0 - 1e-12
        return min(k - 1, int(x * k))

    def float_for_family(self, family, draw=None):
        """A float that decodes to some product WITHIN this family.

        Her fix (2026-08-12) for the natal seeding. The old per-level
        encoder pinned the first level and left the later floats random,
        so a natal agent got a RANDOM PRODUCT inside the family it was
        seeded with. Under the single-float encoding that intent has to
        be written differently: pin the family's interval and leave the
        finer digits of the number free, which is what carries the
        deeper levels. Encoding the family NAME instead would land in
        that level's stop slice and decode as "anything in this family",
        a family-level want rather than a product.

        Returns None if the family is not a top-level branch.
        """
        w = self._stop_width(True)
        choices = self.weights_for_level('ontology', kinds=('product',))
        if not any(choices.values()):
            return None
        cum = self.normalized_cumulative(choices)
        lo = 0.0
        for name, hi in cum.items():
            if name == family:
                a = w + lo * (1.0 - w)
                b = w + hi * (1.0 - w)
                r = random.uniform(a, b) if draw is None else float(draw)
                return min(max(r, a), b - 1e-12)
            lo = hi
        return None

    def float_for_want_block(self, index, existing=None):
        """Encode a want-block index, keeping a float that already points
        there - the same preservation rule as the item encoder."""
        k = self.want_block_count()
        idx = max(0, min(k - 1, int(index)))
        if existing is not None and self.want_block(existing) == idx:
            return float(existing)
        return (idx + 0.5) / k

    def trade_size(self):
        # HER INSTRUCTION (2026-08-13): "dont even have the seller ask for
        # specific items just a test of the item", and "if it makes sense
        # you can share floats as cma es can probably take it".
        #
        # It makes sense only BECAUSE of the first half. A buy that asks
        # with its test has no use for an item; a sell or a construct
        # never seeks, so it has no use for a want pointer. Neither type
        # ever needs both, so ONE slot serves both, read according to the
        # type in the slot before it. A plan is four floats, the genome 88
        # rather than 94.
        #
        # The cost she overrode, recorded so it is not lost: the type
        # bands are stop [0,0.1), buy [0.1,0.4), construct [0.4,0.7),
        # sell [0.7,1.0), so buy and construct are ADJACENT and a small
        # mutation across 0.4 used to turn "build this" into "buy this"
        # with the item carried over intact. With the slot shared, that
        # mutation also reinterprets the slot, so the target is not
        # preserved across the switch.
        # POOLING OFF (her order, 2026-08-13): wave 16's layout, where every
        # plan carries its OWN want inline - a sought sign and four test
        # selectors - instead of a one-float pointer into a shared pool.
        # This exists so a reproduction run can match wave 16's genome
        # rather than only its parameters: restoring item_size to 8 and
        # test_path_size to 4 got the paths right but left a29's pooling in
        # place, and the genome came out 196 floats against wave 16's 226,
        # with three shared wants where wave 16 had six private ones.
        # Whether a plan owns its want or points at a shared one changes
        # what a mutation does and how many distinct wants an agent can
        # hold, which is not a difference a reproduction can carry.
        if not self.pooled():
            return (1 + self.p['item_size'] + 2
                    + self.p['sign_size'] + 4)
        # type 1 + item path + price 2 + want pointer 1 (+ for a Cobb-Douglas human, the good)
        return 1 + self.p['item_size'] + 2 + 1 + (1 if self.is_cd_human() else 0)

    # THE CORPUS TAIL (her ruling 2026-08-21, "lets go with your genome
    # increasing one"). A bank entry held ONE corpus slot, so a joining
    # instruction - data_fusion_concat2 takes two corpora, concat3 takes
    # three - arrived in the program with nothing after it to work on.
    # Measured: 30,845 auditions on a fusion corpus, 100.0% scoring
    # exactly zero, 16.3% of all evaluation.
    #
    # An entry now carries a HEAD slot, which may hold a corpus or a
    # joining instruction, and TAIL slots which may hold only corpora.
    # The head consumes as many tail slots as it needs; a plain corpus
    # consumes none and leaves them unexpressed, which is ordinary
    # non-coding sequence in this encoding.
    #
    # WHY SLOTS RATHER THAN WALKING ONE FLOAT DEEPER (her objection, and
    # it is decisive): restricting a walk by what the operator above it
    # eats would make a corpus's interval depend on WHICH operator and
    # WHICH argument position it sits in, so the encoder would need a
    # separate inverse per context and any disagreement would silently
    # turn an authored corpus into a different one. A tail slot is
    # corpora-only ALWAYS, whatever the head turns out to be, so the rule
    # is a fixed property of the slot and the two directions stay
    # symmetric.
    def data_tail_size(self):
        return int(self.p.get('data_tail_size', 3))

    def offers_a_terminal(self, adict, depth=0):
        """True if anything under this branch takes no inputs."""
        if not isinstance(adict, dict) or depth > 8:
            return False
        args = adict.get('_args')
        kids = [(k, v) for k, v in adict.items()
                if not k.startswith('_') and isinstance(v, dict)
                and '_weight' in v]
        if not kids:
            return isinstance(args, (list, tuple)) and len(args) == 0
        if isinstance(args, (list, tuple)) and len(args) == 0:
            return True
        return any(self.offers_a_terminal(v, depth + 1) for _, v in kids)

    def bank_entry_size(self):
        # test item path + test data path; no stop (fixed length needs
        # no terminator - her correction), no hidden slot (hidden is
        # the LAW, not a flag). THE THRESHOLD RETURNS (her revival,
        # 2026-08-20): under test_threshold worlds each entry carries
        # one more float - the evolved quality bar. Under score-
        # proportional pay the old pathologies are gone: a zero bar is
        # a viable state (junk pays little anyway), so evolution can
        # raise bars wherever waiting pays. The float is NEVER pinned
        # by masks, so human patience evolves too, taught by utility.
        return (2 + self.data_tail_size()) * self.p['test_path_size'] \
            + (1 if self.p.get('test_threshold') else 0)

    def selector(self, afloat):
        # One test pick. HER RULING (2026-08-10): "making half of each
        # of the test pointers go to no test" - selector_none_share
        # 0.5. Natal consequence, stated for the record: with four
        # pointers a natal buy is zero-test 6.25% of the time with 2.0
        # active tests on average (v14's old layout booted at 46.3%
        # zero-test, mean 1.0 - the untested bootstrap lane is thinner
        # now; the iteration-300 early-debug checks watch whether
        # settles still flow). The non-none mass splits uniformly over
        # the bank entries; evolution can still reach any pole.
        none_share = float(self.p.get('selector_none_share', 0.5))
        f = min(max(float(afloat), 0.0), 0.9999999)
        if f < none_share:
            return None
        n = self.p['test_bank_size']
        return int((f - none_share) / (1.0 - none_share) * n)

    def float_for_selector(self, entry_index):
        # Interval midpoint for a pick; None pins the no-test interval.
        none_share = float(self.p.get('selector_none_share', 0.5))
        if entry_index is None:
            return none_share / 2.0
        n = self.p['test_bank_size']
        return none_share + (int(entry_index) + 0.5) * (1.0 - none_share) / n

    def trade_type(self, afloat):
        trade_type = 'stop'

        weighted_choices = OrderedDict([('buy', 1), ('construct', 1), ('sell', 1)])
        #OrderedDict needed if you are not using python 3.7 or above!
        #weighted_choices = {'buy': 1, 'construct': 1, 'sell': 1}
        previously_taken_space = self.p['chance_of_stop_codon']
        choice = self.choose_weighted(weighted_choices, afloat, previously_taken_space=previously_taken_space)
        if choice:
            trade_type = choice
        return trade_type


    def float_for_trade_type(self, trade_type, existing=None):
        # a24 (her order 2026-08-12, "preserve if you recommend it" - I do):
        # if the float already there decodes to this trade type, KEEP IT.
        # The old behaviour drew a fresh point inside the correct slice,
        # which is the same reroll defect she had me remove from the item
        # encoder. Harmless while a trade type is a single flat choice,
        # but it rewrites a genome position for no reason, and the genome
        # is the optimizer's own vector rather than scratch space.
        if existing is not None:
            try:
                if self.trade_type(float(existing)) == trade_type:
                    return float(existing)
            except Exception:
                pass
        float_for_trade_type = random.uniform(self.p['chance_of_stop_codon'],1.0)

        weighted_choices = OrderedDict([('buy', 1),('construct', 1), ('sell', 1)])
        #weighted_choices = {'buy': 1, 'construct': 1, 'sell': 1}
        float_for_choice = self.float_for_weighted_choice(weighted_choices, trade_type, previously_taken_space=self.p['chance_of_stop_codon'])
        if float_for_choice != None:
            float_for_trade_type = float_for_choice
        return float_for_trade_type

    # a25 (her order 2026-08-12, "remove"): the hidden() / float_for_hidden()
    # and stop() / float_for_stop() True-False pairs were deleted here. They
    # had no callers anywhere - hidden became an unconditional law rather
    # than a per-trade choice, and the trade-type 'stop' is a different
    # thing entirely - so they were dead weight that made
    # float_for_weighted_choice look more entangled than it is.

    # HER INVARIANTS (2026-08-20): "sell only names a prodcuct, test only
    # performs a test on data that is a data." The ontology root holds the
    # product families as SIBLINGS of `test` and `data`, all at equal
    # weight, so an item float could decode to `data_freetext_trumpTweets`
    # or `test_classifier_f1heldout` - a want to buy a dataset, or an offer
    # to sell a test. Measured before this gate: 25% of the want range and
    # 23% of sell offers were non-products.
    #
    # A branch declares its kind with `_kind`, alongside the existing
    # `_weight` / `_args` / `_deterministic` metadata, so no path name
    # changes and the registry is untouched. An ontology that has not been
    # annotated yet falls back to the structural reading - `test` and
    # `data` are those kinds, everything else is a product - so existing
    # configs keep working and an annotated one overrides the guess.
    _KIND_BY_NAME = {'test': 'test', 'data': 'data'}

    def kind_of_branch(self, name, adict):
        if isinstance(adict, dict) and isinstance(adict.get('_kind'), str):
            return adict['_kind']
        return self._KIND_BY_NAME.get(name, 'product')

    def weights_for_level(self, cumulative_category, kinds=None,
                          terminals_only=False):
        weights_for_level = OrderedDict()
        if not cumulative_category.endswith('_stop') and not cumulative_category == 'stop':
            cat_dict = self.retrieve_ontology_item(cumulative_category)
            at_root = (cumulative_category == 'ontology')
            for name, adict in cat_dict.items():
                if not name.startswith('_') and isinstance(adict, dict) and '_weight' in adict:
                    if (kinds is not None and at_root
                            and self.kind_of_branch(name, adict) not in kinds):
                        continue        # wrong kind of thing for this slot
                    if terminals_only and not self.offers_a_terminal(adict):
                        continue        # a tail slot holds corpora only
                    weights_for_level[name] = adict['_weight']

        return weights_for_level

    # ------------------------------------------------------------------
    # a21: ONE FLOAT PER ONTOLOGY REFERENCE (her design, 2026-08-12)
    #
    #   "instead of 8 floats, each ontology specification has one float,
    #    such that the lower resolution end of the single float is the
    #    root and the higher resolution end of the single float is the
    #    branches of the tree ... the stop codon would be zeros in the
    #    higher resolution parts of the float."
    #
    # The leading part of the number names the family, the next part the
    # sub-family, the finest part the exact product. At every level the
    # LOW slice, of width chance_of_stop_codon, means STOP - which is
    # why it has to be a slice and not the point zero: a continuous
    # number never lands exactly on a point, and index zero would
    # otherwise collide with the first child. A number that stops early
    # is a family-level reference; a leading stop is a reference to
    # anything at all, which for a learner's want means "any want
    # whatsoever" (humans are pinned and never reach it).
    #
    # Everything below a stop is READ NO FURTHER AND NEVER ALTERED - her
    # correction: "not blanked in the algorithm but interpreted as
    # blanked". The tail belongs to the genome, which is the optimizer's
    # own vector, not scratch space.
    #
    # NOTE the convention change this makes explicit. In the old
    # per-level decode the stop slice sat at the TOP of each interval
    # (choose_weighted returns None for a high roll), so high values
    # meant stop. Her rule is zeros, so here it sits at the bottom.
    # ------------------------------------------------------------------

    def _stop_width(self, include_stop=True):
        return float(self.p['chance_of_stop_codon']) if include_stop else 0.0

    def _stop_allowed(self, cumulative_category):
        """Stops are offered at EVERY level, in both representations.

        HER CLARIFICATION (2026-08-14) of the 2026-08-13 "Do 1 remove
        stop" ruling, which this method used to implement as a grammar
        restriction in the single-float encoder only: "you misunderstood
        what i said. a seller can not sell a vectorSpace_stop is what I
        meant." The ruling is a fact about the MARKET, not about the
        language. An offer whose name has no implementation translates
        to nothing, earns nothing, and is selected against - the same
        as offering a bare stop. It costs auditions, and wave 16 paid
        that cost its whole life (about one sell listing in six) while
        complexifying better than anything since. A want that names a
        family ("any vectorSpace") stays meaningful.

        Restricting the grammar instead had two effects she rejected:
        the two representations no longer described the same set of
        names, and the authored human wants - written as
        `vectorSpace_stop`, "any vector space", judged by
        `test_vectorSpace_probe` - could not be expressed in the
        single-float arm at all.
        """
        return True

    def ontology_item_single(self, afloat, category='ontology',
                             include_stop=True, max_depth=64, kinds=None,
                             terminals_only=False):
        """Decode ONE float into an ontology path, coarse to fine.

        """
        cumulative_category = category
        x = float(afloat)
        if x < 0.0:
            x = 0.0
        if x >= 1.0:
            x = 1.0 - 1e-12
        for _ in range(max_depth):
            # Per level now, not once: a stop is only offered where it
            # names something runnable (her 2026-08-13 ruling).
            w = (self._stop_width(include_stop)
                 if self._stop_allowed(cumulative_category) else 0.0)
            _names, _tops, _anyw = self._level_table(cumulative_category, kinds, terminals_only)
            if not _anyw:
                # A leaf. Stochastic leaves still take their copy index.
                if self.parameters_set(cumulative_category) \
                        and self.is_stochastic(cumulative_category):
                    cumulative_category = (cumulative_category
                                           + "_stochastic"
                                           + str(self.stochastic_roll()))
                break
            if x < w:                       # the STOP slice, at the low end
                cumulative_category = ('stop' if cumulative_category == 'ontology'
                                       else cumulative_category + "_stop")
                break
            x = (x - w) / (1.0 - w)         # rescale past the stop slice
            # The level's cumulative table is computed once per run and searched by bisection (hard keys,
            # 5 Oct 2026: a node may have tens of thousands of children). Same floats, same order, same result
            # as the linear scan it replaces: the first child whose top exceeds x.
            _i = bisect.bisect_right(_tops, x)
            if _i < len(_tops):
                chosen, hi = _names[_i], _tops[_i]
                lo = _tops[_i - 1] if _i > 0 else 0.0
            else:                           # numerical edge: take the last
                chosen, hi = _names[-1], 1.0
                lo = _tops[-1]
            span = hi - lo
            x = (x - lo) / span if span > 0 else 0.0
            cumulative_category = (chosen if cumulative_category == 'ontology'
                                   else cumulative_category + "_" + chosen)
        return cumulative_category

    def float_for_ontology_item_single(self, ontology_item, include_stop=True,
                                       existing=None, category='ontology',
                                       kinds=None):
        """Encode a path into ONE float.

        If `existing` already decodes to this item it is KEPT UNCHANGED -
        her rule, and the fix for the encoder that used to reroll a float
        that already meant the right thing. Under this encoding the finer
        digits ARE the deeper branches, so rerolling them destroys the
        latent tail.

        Otherwise the value is the midpoint of the interval the path
        names - deterministic, no draw.
        """
        if existing is not None:
            try:
                if self.ontology_item_single(existing, category=category,
                                             include_stop=include_stop,
                                             kinds=kinds) == ontology_item:
                    return float(existing)
            except Exception:
                pass
        # w is per level now, computed from `cur` inside the loop, so the
        # encoder reserves a stop slice exactly where the decoder does.
        lo, hi = 0.0, 1.0
        cur = category
        parts = [p for p in str(ontology_item).split('_') if p]
        if category != 'ontology':
            # The float is read from inside a subtree (a test hangs under
            # test_<kind>, a data set under data), so the levels the
            # category already names are not in the number.
            _pre = [p for p in str(category).split('_') if p]
            if parts[:len(_pre)] == _pre:
                parts = parts[len(_pre):]
        stopped = False
        for part in parts:
            if part.startswith('stochastic'):
                break                       # the copy index is not in the path
            w = (self._stop_width(include_stop)
                 if self._stop_allowed(cur) else 0.0)
            span = hi - lo
            if part == 'stop':
                if w <= 0.0:
                    return None             # no stop is offered at this level
                hi = lo + w * span          # land inside this level's stop slice
                stopped = True
                break
            choices = self.weights_for_level(cur, kinds=kinds)
            if not any(choices.values()):
                break                       # already at a leaf
            base = lo + w * span
            inner = hi - base
            cum = self.normalized_cumulative(choices)
            clo, chi, found = 0.0, None, False
            for name, top in cum.items():
                if name == part:
                    chi, found = top, True
                    break
                clo = top
            if not found:
                return None                 # not a real path
            lo, hi = base + clo * inner, base + chi * inner
            cur = part if cur == 'ontology' else cur + "_" + part
        if not stopped and any(self.weights_for_level(cur, kinds=kinds).values()):
            # The path named an interior node without saying "stop". Put
            # it in that level's stop slice so it decodes back the same
            # way - but only where a stop is offered at all. Since her
            # 2026-08-13 ruling it usually is not, and then the item is
            # simply not expressible: say so, rather than return a float
            # that decodes to something else.
            _w = (self._stop_width(include_stop)
                  if self._stop_allowed(cur) else 0.0)
            if _w <= 0.0:
                return None
            span = hi - lo
            hi = lo + _w * span
        return (lo + hi) / 2.0

    def parameters_set(self,cumulative_category):
        parameters_set = False
        #check if this category is at the last level, whether it has a stop condon or not
        if cumulative_category.endswith('_stop'):
            cumulative_category = cumulative_category[:-5]
        if not cumulative_category == 'stop' and not self.weights_for_level(cumulative_category):
            parameters_set = True

        return parameters_set

    def is_stochastic(self,cumulative_category):
        description = self.retrieve_ontology_item(cumulative_category)
        is_stochastic = not description['_deterministic'] if '_deterministic' in description else False
        return is_stochastic

    def stochastic_roll(self):
        n = self.p['stochastic_copies']
        weighted_choices = OrderedDict([(x,1) for x in range(n)])
        roll = random.uniform(0, 1)
        stochastic_roll = self.choose_weighted(weighted_choices,roll)
        return stochastic_roll


    def ontology_item(self, float_vec, category='ontology', include_stop=True,
                      kinds=None, terminals_only=False):
        # if category is filled in, the float starts from the given category
        cumulative_category = category
        #print ('in ontology item , category')
        #print (category)
        for afloat in float_vec:
            weighted_choices = self.weights_for_level(cumulative_category, kinds=kinds,
                                                      terminals_only=terminals_only)
            #print('weighted_choices')
            #print(weighted_choices)

            if not any(weighted_choices.values()):
                #You have come to the end of what is determined.
                #Now see if that is stochastic
                if self.parameters_set(cumulative_category) and self.is_stochastic(cumulative_category):
                    cumulative_category = cumulative_category + "_stochastic" + str(self.stochastic_roll())

                break
            # roll = random.uniform(0, 1)
            roll = afloat
            #print ('afloat')
            #print (afloat)
            if include_stop:
                choice = 'stop'
                previously_taken_space = self.p['chance_of_stop_codon']
            else:
                previously_taken_space = 0

            #print('previously_taken_space')
            #print(previously_taken_space)

            guess = self.choose_weighted(weighted_choices, roll, previously_taken_space=previously_taken_space)
            #print('guess')
            #print(guess)
            if guess:
                choice = guess
            if cumulative_category == 'ontology':
                cumulative_category = choice
            else:
                cumulative_category = cumulative_category + "_" + choice

        return cumulative_category




    def floats_for_ontology_item(self, ontology_item, include_stop=True, skip = 0,
                                 kinds=None, terminals_only=False):
        # if category is filled in, the float starts from the given category
        #print("floats_for_ontology_item")
        #print ('ontology_item')
        #print (ontology_item)
        float_list = []
        category_list = ontology_item.split("_")

        cumulative_category = "ontology"
        for choice in category_list:


            #print('cumulative_category')
            #print(cumulative_category)

            weighted_choices = self.weights_for_level(cumulative_category, kinds=kinds,
                                                      terminals_only=terminals_only)
            #print('weighted_choices')
            #print(weighted_choices)


            if include_stop:
                previously_taken_space = self.p['chance_of_stop_codon']
            else:
                previously_taken_space = 0

            #print('previously_taken_space')
            #print(previously_taken_space)

            float_guess = self.float_for_weighted_choice(weighted_choices, choice, previously_taken_space)
            if float_guess == None:
                float_guess = random.uniform(self.p['chance_of_stop_codon'], 1.0)
            float_list.append(float_guess)
            #print('float_guess')
            #print(float_guess)

            if cumulative_category == 'ontology':
                cumulative_category = choice
            else:
                cumulative_category = cumulative_category + "_" + choice

        # HER RULING (2026-08-12): "that is a bug, and you should remove
        # it from the code in case we revert."
        #
        # What was here: the positions beyond this item's path were
        # filled with FRESH RANDOM VALUES, and `skip` popped entries off
        # the front while appending more random values at the back to
        # hold the length. A float that is not part of an item's path
        # must be left alone - it belongs to the genome, and the genome
        # is the optimizer's own vector.
        #
        # In one caller the padding was merely wasted (the item encoder
        # copies only the first len(path) values, so the randoms were
        # generated and discarded - though they still consumed draws
        # from the seeded stream). In the TEST BANK it was live
        # corruption: those callers pass skip=2 or skip=1 and then copy
        # the first test_path_size floats, so for a three-level test
        # name one real float and THREE RANDOM ONES were written into
        # the genome for every bank entry, every time an authored plan
        # was encoded.
        #
        # Now: return only the floats that encode the path. Callers copy
        # that prefix and leave the rest of the vector exactly as the
        # optimizer made it. This is the same principle as reading an
        # item only to its stop and never altering the tail.
        for _ in range(skip):
            if float_list:
                float_list.pop(0)

        #print ('len(float_list)')
        #print (len(float_list))
        #print ('self.p[item_size]')
        #print (self.p['item_size'])
        #print ('float_list')
        #print (float_list)
        return float_list

    def float_for_agi_token(self, tokens):
        afloat = (tokens - self.p["min_token_price"])/(self.p["max_token_price"] - self.p["min_token_price"])
        if afloat > 1.0:
            afloat = 1.0
        elif afloat < 0.0:
            afloat = 0.0
        return afloat

    def agi_token(self, afloat):
        return self.p["min_token_price"] + (self.p["max_token_price"] - self.p["min_token_price"]) * afloat

    def float_for_weighted_choice(self, weighted_choices, atype, previously_taken_space=0):
        # reverse of choose weighted function
        # return null if roll returns within previously taken space

        cumulative = self.normalized_cumulative(weighted_choices)
        space = 1 - previously_taken_space

        choice_float = None
        last_weight = 0.0
        for choice, weight in cumulative.items():
            if choice == atype:
                choice_float = random.uniform(last_weight*space, weight*space)
            last_weight = weight
        if choice_float == None:
            choice_float = random.uniform(last_weight*space, 1.0)
        return choice_float


    def _level_table(self, cat, kinds, terminals_only):
        """(child names, cumulative tops, any weight) for one ontology level, computed once per run with
        weights_for_level and normalized_cumulative (the ontology is fixed for the whole run)."""
        cache = getattr(self.model, '_ont_level_cache', None)
        if cache is None:
            cache = self.model._ont_level_cache = {}
        key = (cat, tuple(kinds) if kinds is not None else None, bool(terminals_only))
        hit = cache.get(key)
        if hit is None:
            choices = self.weights_for_level(cat, kinds=kinds, terminals_only=terminals_only)
            anyw = any(choices.values())
            names, tops = [], []
            if anyw:
                cum = self.normalized_cumulative(choices)
                names, tops = list(cum.keys()), list(cum.values())
            hit = cache[key] = (names, tops, anyw)
        return hit

    def normalized_cumulative(self, weighted_choices):

        # transform to cumulative distribution
        total = 0
        for choice, weight in weighted_choices.items():
            total += weight

        normalized = OrderedDict([(choice, (weight / total)) for choice, weight in weighted_choices.items()])

        previous_weight = 0
        cumulative = OrderedDict()
        for choice, weight in normalized.items():
            cumulative[choice] = weight + previous_weight
            previous_weight += weight

        return cumulative


    def choose_weighted(self, weighted_choices, roll, previously_taken_space=0):
        # return null if roll returns within previously taken space
        cumulative = self.normalized_cumulative(weighted_choices)
        space = 1 - previously_taken_space
        chosen = None

        for choice, weight in cumulative.items():
            if roll < weight * space:
                chosen = choice
                break


        return chosen

    def convert_to_cumulative_category(self, function_name, call_depth=0):
        #this could either be a ontology item already, or an emergent function
        #it its emergent, recursively take the first item until a
        #cumulative category is reached, within the recursion limit

        name = function_name
        if name in self.model.emergent_functions:
            chain = self.model.emergent_functions[name]
            if not chain:
                # Marked-empty chain (unresolvable placeholder, the
                # 2026-05-21 graceful-failure convention): no root to
                # descend into. Return the name unchanged, like any
                # non-emergent token; the offer dies at the
                # `if itemlist:` guards, never here. Unguarded, this
                # IndexError killed every run at its first such query
                # (the cache-frontier deaths, closed 2026-08-04).
                return name
            emergent_root = chain[0]
            if call_depth < self.p["recursive_trade_depth_limit"]:
                name = self.convert_to_cumulative_category(emergent_root, call_depth = call_depth+1)
        return name


    def item_type(self, general_function):
        cumulative_category = self.convert_to_cumulative_category(general_function)
        item_type = self.item_type_pattern.search(cumulative_category)
        if (item_type):
            item_type = item_type.group(1)
        return(item_type)


    def test_item_type(self, cumulative_category):
        item_type = self.test_item_type_pattern.search(cumulative_category)
        if (item_type):
            item_type = item_type.group(1)
        return(item_type)


    def float_vec_to_trade_plan(self, float_vec_dont_change, mask = None):
        # Genome version 3 (the curator 2026-08-10). Layout: display sign |
        # six plans of (type, sign, item, price, two selectors) | a
        # shared TEST BANK of test_bank_size entries, each a test item
        # path and a test data path of test_path_size floats. Tests
        # exist only on buys, resolved from the bank through the
        # selectors; a bank entry's floats are decoded UNDER THE
        # BUYING PLAN's item type (the typed-test law), so one entry
        # can serve differently-typed buys. Masks (the authored wants)
        # pin by presence exactly as before; a mask trade's authored
        # tests are planted into bank entries allocated in order, and
        # its selectors pinned at them, so the authored demand is
        # preserved verbatim while everything unpinned evolves.

        float_vec = copy.deepcopy(float_vec_dont_change)

        first_level = ["distributes", "initial_message", "final_message", "message_period"]

        trade_plan = OrderedDict([('type', self.__class__.__name__)])
        trade_plan['label'] = trade_plan['type'] + " Agent " + str(self.unique_id)
        if mask and "label" in mask:
            trade_plan['label'] = mask["label"] + ", " + trade_plan['label']
        for name in first_level:
            if mask and name in mask:
                trade_plan[name] = mask[name]

        cursor = 0
        next_cursor = self.p['sign_size']
        trade_plan['sign'] = list(float_vec[cursor:next_cursor])
        if mask and "sign" in mask:
            for i in range(min(len(mask["sign"]), self.p['sign_size'])):
                trade_plan['sign'][i] = mask["sign"][i]
                float_vec[cursor + i] = mask["sign"][i]
        if self.p.get('sign_space_full'):
            # HER GRADIENT LEVER 1 (2026-08-17): signs live on [-1,1]
            # (the LLM substrate's written law; the [0,1] confinement
            # was the July 14 parity bug). Applied at emission only -
            # genome space stays [0,1]; the message speaks the full
            # space, opposition expressible, random pairs near 90 deg.
            trade_plan['sign'] = [2.0 * float(x) - 1.0
                                  for x in trade_plan['sign']]
        cursor = next_cursor

        bank_offset = self.p['sign_size'] \
            + self.n_plans() * self.trade_size()
        tp_size = self.p['test_path_size']
        bank_alloc = 0          # next free bank entry for mask pins
        pinned_tests = {}       # plan index -> authored test dicts

        trade_plan['trades'] = []
        cursor_before_trade_plans = cursor

        for i in range(self.n_plans()):
            trade_plan['trades'].append(dict())
            cursor = cursor_before_trade_plans + i * self.trade_size()
            m_i = (mask['trades'][i] if mask and "trades" in mask
                   and i < len(mask['trades']) else None)

            # type
            if m_i and 'type' in m_i:
                trade_plan['trades'][i]['type'] = m_i['type']
                float_vec[cursor] = self.float_for_trade_type(
                    m_i['type'], existing=float_vec[cursor])
            else:
                trade_plan['trades'][i]['type'] = self.trade_type(float_vec[cursor])
            cursor += 1

            # THE WANT IS EXPRESSED, AND THE TEST STARTS FROM IT.
            # HER RULING (2026-08-20): ask_with_test_only is gone, and both
            # humans and learners return to the April 4 order - the buy
            # decodes its ITEM first, its family is taken from that item,
            # and the test decodes from inside test_<that family>. Under
            # test-only the derivation ran the other way: the test decoded
            # from the whole test tree and the item was back-filled from
            # whatever family the test named, which welded the want and the
            # standard onto a single float (item_size is 1 here), so a buy
            # could not refine how it judges without also changing what it
            # asks for. That is the coordinate a coevolving market most
            # needs to move independently. April 4: commit 141891f,
            # MarketAgent.py line 1251, category = 'test_' + item_type.

            # item
            next_cursor = cursor + self.p['item_size']
            if self.p.get('single_float_ontology', True):
                # a21: the whole path lives in ONE float, read coarse to
                # fine. An injected item keeps the float already there if
                # it already decodes to that item, so the finer digits -
                # the deeper branches - survive.
                # HEAD AND TAIL (her instruction 2026-08-12, "have
                # sufficient room, and have a stop codon" - then "examine
                # the gep rules that parse the program").
                #
                # Gene expression programming sizes a gene as a HEAD of
                # length h, where functions may appear, and a TAIL of
                # length t = h * (n - 1) + 1 where n is the largest arity
                # in the function set. The tail holds terminals only, and
                # it is sized so that whatever the head opens the tail can
                # always close. Running out of material is impossible by
                # construction rather than by luck.
                #
                # This port had no head-and-tail at all: any plan could
                # hold any product, so a three-input combiner could land
                # in the last position with nothing behind it. In the
                # first eight boards of a branch-capable world, 9,548 of
                # 29,138 combiner calls arrived starved.
                #
                # NO HEAD AND TAIL. Her correction (2026-08-12):
                # "why do you need head and tail lines if they can buy
                # from each other. i think the original gep worked with
                # arity, and if it didnt, the bug was small and didnt
                # require restructuring."
                #
                # She is right and I had restructured around a small bug.
                # The original mechanism already handles an incomplete
                # expression: its deficit IS its arity, so it is a PART,
                # and a buyer splices it in and supplies what it lacks -
                # the market does the composition, which is the point of
                # the system. The only real fault was RUNNING an
                # unfinished expression, which is two lines in
                # perform_test.
                if m_i and 'item' in m_i and m_i['item'] is not None:
                    _f = self.float_for_ontology_item_single(
                        m_i['item'], existing=float_vec[cursor],
                        kinds=('product',))
                    if _f is None:
                        # HER ORDER (2026-08-14): no silent failure - see
                        # the matching raise in the shared-slot branch.
                        raise ValueError(
                            'authored item %r cannot be written in the '
                            'single-float encoding'
                            % (m_i['item'],))
                    float_vec[cursor] = _f
                trade_plan['trades'][i]['item'] = \
                    self.ontology_item_single(float_vec[cursor],
                                              kinds=('product',))
                if self.p.get('seller_name_defaults') \
                        and trade_plan['trades'][i]['type'] in ('sell', 'construct'):
                    _it2 = trade_plan['trades'][i]['item']
                    if _it2 != 'stop' and 'stop' in str(_it2).split('_'):
                        # HER RULING (5 Oct 2026, point 2): "decoder_stop in a buyer means a decoder,
                        # and decoder_stop in a seller means a designated one ... designated in the
                        # ontology". The genome is not touched; the name it decodes to is kept in
                        # _named, and the row sells the designated default below it. A bare stop is
                        # left as it is (it also ends a section in parts of the engine).
                        trade_plan['trades'][i]['_named'] = _it2
                        trade_plan['trades'][i]['item'] = self.default_leaf(_it2)
            else:
                if m_i and 'item' in m_i:
                    floats_for_item = self.floats_for_ontology_item(
                        m_i['item'], kinds=('product',))
                    ontlist = m_i['item'].split("_")
                    for k in range(min(len(ontlist), self.p['item_size'])):
                        float_vec[cursor + k] = floats_for_item[k]
                trade_plan['trades'][i]['item'] = self.ontology_item(
                    float_vec[cursor:next_cursor], kinds=('product',))
                if self.p.get('seller_name_defaults') \
                        and trade_plan['trades'][i]['type'] in ('sell', 'construct'):
                    # her 5 Oct point 2 under the many-numbers name as well (6 Oct night, her point that
                    # "close on ontology should be close in distance on the float" and that the one-number name
                    # has no gradient at depth), so the two encodings differ in nothing but the encoding
                    _it6 = trade_plan['trades'][i]['item']
                    if _it6 != 'stop' and 'stop' in str(_it6).split('_'):
                        trade_plan['trades'][i]['_named'] = _it6
                        trade_plan['trades'][i]['item'] = self.default_leaf(_it6)
            # FOCUS (6 Oct 2026 night; her "maybe its a capacity issue, like attention or window context size and
            # focus" and her order to design the scenario that works and run it): classes_by_learner, a list cycled
            # by agent number, locks a learner's sell and construct rows to one class of the cipher ontology - it
            # can decode that class and no other - so the KIND of what it sells is a stable trait. The reason (the
            # sign-formation model, DEBUG_LADDER_20261006.md section 12): signs form in 2,000-8,000 boards when every
            # seller's kind is fixed and never when sellers change kind even 1% per generation, and in the ladder
            # worlds a learner's section changed kind on two of three ten-board steps. Buy rows are untouched: a
            # specialist may buy any class. Absent: nothing changes. The genome is not touched; the name the row
            # decoded to is kept in _named.
            _cbl = self.p.get('classes_by_learner')
            if _cbl and trade_plan['trades'][i]['type'] in ('sell', 'construct') and not self.is_cd_human():
                _it4 = str(trade_plan['trades'][i]['item'])
                _segs4 = _it4.split('_')
                if len(_segs4) >= 3 and _segs4[0] == 'decoder' and re.match(r'^c\d+$', _segs4[2]):
                    _lock4 = 'c%d' % int(_cbl[int(self.unique_id) % len(_cbl)])
                    if _segs4[2] != _lock4:
                        trade_plan['trades'][i].setdefault('_named', _it4)
                        _segs4[2] = _lock4
                        trade_plan['trades'][i]['item'] = '_'.join(_segs4)
            cursor = next_cursor

            # price midpoint and range (range never pinned - her law)
            if m_i and 'midpoint' in m_i:
                trade_plan['trades'][i]['midpoint'] = m_i['midpoint']
                float_vec[cursor] = self.float_for_agi_token(m_i['midpoint'])
            else:
                trade_plan['trades'][i]['midpoint'] = self.agi_token(float_vec[cursor])
            cursor += 1
            if m_i and 'range' in m_i:
                trade_plan['trades'][i]['range'] = m_i['range']
                float_vec[cursor] = self.float_for_agi_token(m_i['range'])
            else:
                trade_plan['trades'][i]['range'] = self.agi_token(float_vec[cursor])
            cursor += 1

            # THE WANT POINTER (her design, 2026-08-12). One float says
            # WHICH standing want this plan expresses. EVERY row reads
            # it (her ruling 2026-09-11): a buy seeks a seller with it,
            # a sell seeks a buyer with it - so one agent can seek up to
            # three different kinds of buyer, one per want block, and
            # buyers_weight below 1.0 lets the seller side count in
            # gather_offers. Until today a sell row's sign was a paste
            # of the displayed sign, which made every buyers_weight but
            # 1.0 compare display against display - noise.
            if not self.pooled():
                # Wave 16's layout: the want is INSIDE this plan - a sought
                # sign then four test selectors - so there is no pointer and
                # no shared pool. Everything below reads through _wb_off and
                # _sel_off, so pointing them into the plan is the whole
                # change; no reader needs to know which layout it is.
                _wb_off = cursor
                _sel_off = _wb_off + self.p['sign_size']
                trade_plan['trades'][i]['_want'] = None
                cursor += self.p['sign_size'] + 4
            elif self.is_cd_human() and (self.p.get('humans_cd') or {}).get('good_tree'):
                # HER DECISION 18 (6 Oct 2026, 23:3x, "18 yes"): the human row's good and its want as a TREE of
                # numbers with wide slices. Before, the good was one number cut into as many slices as goods (7, so
                # the row asked for a different good on about half the boards at step 0.1) and the want was a
                # second 7-slice pointer read on its own. Now the first number picks the rung (singles, pairs, the
                # whole: as many slices as rungs) and the second the good within the rung (at most four slices),
                # and the want IS the good's own standing want, so there is no separate pointer. The genome's
                # layout is unchanged (the same two numbers); only the reading is. "Close on the ontology, close
                # on the number" (her words); the drift of a row's want falls from about half the boards to about
                # one in eight.
                _tree = self.p['humans_cd']['good_tree']
                _goods = self.p['humans_cd']['goods']
                _f1 = min(max(float(float_vec[cursor]), 0.0), 1.0 - 1e-12)
                _f2 = min(max(float(float_vec[cursor + 1]), 0.0), 1.0 - 1e-12)
                _rung = _tree[min(len(_tree) - 1, int(_f1 * len(_tree)))]
                _good_t = _rung[min(len(_rung) - 1, int(_f2 * len(_rung)))]
                _wb = _goods.index(_good_t)
                trade_plan['trades'][i]['_want'] = _wb
                trade_plan['trades'][i]['_cd_good_tree'] = _good_t
                cursor += 2
                _wb_off = self.want_block_offset() + _wb * self.want_block_size()
                _sel_off = _wb_off + self.p['sign_size']
            else:
                _wb = self.want_block(float_vec[cursor])
                trade_plan['trades'][i]['_want'] = _wb
                cursor += 1
                _wb_off = self.want_block_offset() + _wb * self.want_block_size()
                _sel_off = _wb_off + self.p['sign_size']
                if self.is_cd_human():
                    trade_plan['trades'][i]['_cd_choice'] = float(float_vec[cursor])
                    cursor += 1
            # BUYER-ONLY MATCHING (her rulings 11 Aug, 30 Sep, 2 Oct): only a
            # BUY row decodes a sought sign; a sell or construct row carries
            # the agent's displayed sign (the sign is about the good sold).
            # The 11 Sep sells-seek decode is removed.
            if trade_plan['trades'][i]['type'] != 'buy' and not self.p.get('sells_seek'):
                trade_plan['trades'][i]['sign'] = list(trade_plan['sign'])
            elif trade_plan['trades'][i]['type'] != 'buy':
                # SELLS SEEK (flag sells_seek, off unless set; 7 Oct 2026, for her control "buyer and seller weight
                # at .5"): a sell or construct row decodes its OWN sought sign from its sign slot (inline when its
                # wants are not pooled, the pointed want block when they are), as every row did in the original
                # engine, so that buyers_weight below 1.0 blends the seller's seeking with the buyer's instead of
                # comparing display against display. Her rulings of 11 Aug, 30 Sep and 2 Oct (buyer-only) stand
                # wherever the flag is unset.
                trade_plan['trades'][i]['sign'] = list(
                    float_vec[_wb_off:_wb_off + self.p['sign_size']])
                if self.p.get('sign_space_full'):
                    trade_plan['trades'][i]['sign'] = [
                        2.0 * float(x) - 1.0 for x in trade_plan['trades'][i]['sign']]
            else:
                trade_plan['trades'][i]['sign'] = list(
                    float_vec[_wb_off:_wb_off + self.p['sign_size']])
                if m_i and 'sign' in m_i:
                    for j in range(min(len(m_i['sign']), self.p['sign_size'])):
                        trade_plan['trades'][i]['sign'][j] = m_i['sign'][j]
                        float_vec[_wb_off + j] = m_i['sign'][j]
            if trade_plan['trades'][i]['type'] == 'buy' and self.p.get('sign_space_full'):
                # Parity completion (bug fix 2026-08-18, found by
                # her "then debug it"): sought signs stayed [0,1]
                # while displays spoke [-1,1] - two geometries, so
                # buyers could not point at half the sign space and
                # sellers were pulled back into the positive cone.
                # ONE rule for both spaces, as she ruled.
                trade_plan['trades'][i]['sign'] = [
                    2.0 * float(x) - 1.0
                    for x in trade_plan['trades'][i]['sign']]

            # test selectors. An authored mask with tests plants them
            # into the bank. Two modes (her rulings 2026-08-10):
            #   pinned (default): selectors pinned at the authored
            #     entries and the tests pass through whole - discipline
            #     authored, the wave-16 world.
            #   human_tests_learned: the WHAT stays authored (every
            #     bank entry carries an authored test, planted
            #     cyclically, so any non-none selector demands an
            #     authored test and demand can never be evolved away)
            #     but the WHETHER is learned - selectors evolve freely,
            #     so a human can learn credulity or learn verification.
            #     The door to the competing social states.
            if m_i and 'tests' in m_i:
                authored = [t for t in m_i['tests']
                            if t.get('test') and not t.get('stophere')]
                learned = self.p.get('human_tests_learned') and authored
                if learned:
                    for entry in range(self.p['test_bank_size']):
                        t = authored[entry % len(authored)]
                        e_off = bank_offset + entry * self.bank_entry_size()
                        if self.p.get('single_float_ontology', True):
                            # a21: one float per path, encoded from the
                            # category it hangs under, keeping whatever is
                            # already there when it already decodes right.
                            # Plant where the reader reads: the selectors
                            # decode a bank entry from test_<family>, so
                            # the plant must encode from there too.
                            _tcat = '_'.join(str(t['test']).split('_')[:2])
                            _tf = self.float_for_ontology_item_single(
                                t['test'], existing=float_vec[e_off],
                                category=_tcat)
                            if _tf is None:
                                # HER ORDER (2026-08-14): no silent failure.
                                raise ValueError(
                                    'authored test %r cannot be written in '
                                    'the single-float encoding'
                                    % (t['test'],))
                            float_vec[e_off] = _tf
                            _df = self.float_for_ontology_item_single(
                                t['data'], existing=float_vec[e_off + tp_size],
                                category='data')
                            if _df is None:
                                # HER ORDER (2026-08-14): no silent failure.
                                raise ValueError(
                                    'authored test data %r cannot be written '
                                    'in the single-float encoding'
                                    % (t['data'],))
                            float_vec[e_off + tp_size] = _df
                        else:
                            tfl = self.floats_for_ontology_item(
                                t['test'], skip=2)
                            for k in range(min(len(tfl), tp_size)):
                                float_vec[e_off + k] = tfl[k]
                            dfl = self.floats_for_ontology_item(t['data'], skip=1)
                            for k in range(min(len(dfl), tp_size)):
                                float_vec[e_off + tp_size + k] = dfl[k]
                    # selectors NOT pinned, tests NOT passed verbatim
                else:
                    picks = []
                    for t in authored[:4]:
                        if bank_alloc >= self.p['test_bank_size']:
                            break
                        entry = bank_alloc
                        bank_alloc += 1
                        e_off = bank_offset + entry * self.bank_entry_size()
                        if self.p.get('single_float_ontology', True):
                            # a21: one float per path, encoded from the
                            # category it hangs under, keeping whatever is
                            # already there when it already decodes right.
                            # Plant where the reader reads: the selectors
                            # decode a bank entry from test_<family>, so
                            # the plant must encode from there too.
                            _tcat = '_'.join(str(t['test']).split('_')[:2])
                            _tf = self.float_for_ontology_item_single(
                                t['test'], existing=float_vec[e_off],
                                category=_tcat)
                            if _tf is None:
                                # LOUD, NOT FATAL (2026-08-14). Some
                                # authored tests exist only in the registry
                                # and were never in the coding tree - wave
                                # 16 included (test_vectorSpace has only
                                # 'probe'). The buy still receives the
                                # authored test VERBATIM through the pinned
                                # branch, so nothing changes meaning; the
                                # bank entry is simply left as the
                                # optimizer made it, which is what every
                                # era before this one did silently.
                                print('WARNING: authored test %r is not in '
                                      'the coding tree; it reaches its buy '
                                      'verbatim, but no bank plant is '
                                      'written' % (t['test'],))
                            else:
                                float_vec[e_off] = _tf
                            _df = self.float_for_ontology_item_single(
                                t['data'], existing=float_vec[e_off + tp_size],
                                category='data')
                            if _df is None:
                                print('WARNING: authored test data %r is '
                                      'not in the coding tree; it reaches '
                                      'its buy verbatim, but no bank plant '
                                      'is written' % (t['data'],))
                            else:
                                float_vec[e_off + tp_size] = _df
                        else:
                            tfl = self.floats_for_ontology_item(
                                t['test'], skip=2)
                            for k in range(min(len(tfl), tp_size)):
                                float_vec[e_off + k] = tfl[k]
                            dfl = self.floats_for_ontology_item(t['data'], skip=1)
                            for k in range(min(len(dfl), tp_size)):
                                float_vec[e_off + tp_size + k] = dfl[k]
                        picks.append(entry)
                    for slot in range(4):
                        pin = picks[slot] if slot < len(picks) else None
                        float_vec[_sel_off + slot] = self.float_for_selector(pin)
                    pinned_tests[i] = [
                        dict(t, _entry=(picks[j] if j < len(picks)
                                        else None))
                        for j, t in enumerate(authored)]
            # The four test selectors live in the want block too: a want
            # is a sign to seek AND the tests that qualify it, which is
            # why they are pointed at together.
            trade_plan['trades'][i]['_selectors'] = [
                self.selector(float_vec[_sel_off + k]) for k in range(4)]

            # carry through config-declared trade fields the decoder
            # does not itself set (alpha, budget, learned_price, ...)
            if m_i:
                for _k, _v in m_i.items():
                    if _k not in trade_plan['trades'][i] and _k != 'tests':
                        trade_plan['trades'][i][_k] = _v

        # THE LEARNED FOCUS (6 Oct 2026 night; her words: "its supposed to change at first then converge as forces
        # develop that push it in a direction ... what is important is those forces exist" and "the signs focus
        # selective pressures on behaviors, the behaviors focus selective pressure on signs"). With focus_learned set,
        # the learner's FIRST row's item gene names its class (the class level of the cipher ontology, a wide slice),
        # and every sell and construct row decodes that class and only that class. The class is a gene, so it drifts
        # at first and settles under the forces: a crowded class pays each of its sellers less, and a sign read as one
        # class brings that class's draws, which pay only if the bearer decodes it. A first gene that names no class
        # (a stopped name, read before any default) leaves the learner unfocused and its rows decode as they are. Buy
        # rows and humans are untouched. The genome is not touched; a rewritten row keeps its decoded name in _named.
        if self.p.get('focus_learned') and trade_plan['trades'] and not self.is_cd_human():
            _t0 = trade_plan['trades'][0]
            _src5 = str(_t0.get('_named') or _t0.get('item') or '')
            _segs5 = _src5.split('_')
            _focus5 = (_segs5[2] if len(_segs5) >= 3 and _segs5[0] == 'decoder' and re.match(r'^c\d+$', _segs5[2])
                       else None)
            if _focus5 is None and self.p.get('seller_name_defaults'):
                # 22:5x: a first gene that stops before the class is the designated default class (her 5 Oct point 2
                # applied to the focus), never an unfocused generalist. Measured in the first learned-focus worlds: the
                # unfocused could build pairs and wholes alone with one key per class, so they won and spread (6 to 9
                # of 16 by 870 boards), against the scenario's premise that a learner holds one class.
                try:
                    _dk = self.retrieve_ontology_item('decoder')
                    _kind5 = _dk.get('_default') if isinstance(_dk, dict) else None
                    _cls5 = self.retrieve_ontology_item('decoder_' + _kind5).get('_default') if _kind5 else None
                    if _cls5 and re.match(r'^c\d+$', str(_cls5)):
                        _focus5 = str(_cls5)
                except Exception:  # noqa: BLE001
                    _focus5 = None
            trade_plan['_focus'] = _focus5
            if _focus5 is not None:
                for _r5 in trade_plan['trades']:
                    if _r5.get('type') not in ('sell', 'construct'):
                        continue
                    _it5 = str(_r5.get('item'))
                    _sg5 = _it5.split('_')
                    if len(_sg5) >= 3 and _sg5[0] == 'decoder' and re.match(r'^c\d+$', _sg5[2]) \
                            and _sg5[2] != _focus5:
                        _r5.setdefault('_named', _it5)
                        _sg5[2] = _focus5
                        _r5['item'] = '_'.join(_sg5)

        # resolve tests from the bank - buys only (her ruling: do not
        # decode and carry tests where they are not in buys)
        for i in range(self.n_plans()):
            t_i = trade_plan['trades'][i]
            selectors = t_i.pop('_selectors')
            tests = []
            if i in pinned_tests:
                for t in pinned_tests[i]:
                    _thr = 0.0
                    if self.p.get('test_threshold') \
                            and t.get('_entry') is not None:
                        _eo9 = bank_offset \
                            + t['_entry'] * self.bank_entry_size()
                        # HER MAP (2026-08-20): an unlearned float
                        # means NO demand - bar = max(0, 2f-1), so the
                        # bottom half of float space is a zero bar and
                        # pickiness must be EARNED by pushing f up.
                        # F1 (2026-09-10): the bar is the LAST float of the
                        # entry, after the extra corpora - the same offset the
                        # learner path reads at. '2 * tp_size' predated the
                        # corpus tail and read the first tail slot instead.
                        _f9 = float(min(max(
                            float_vec[_eo9 + (2 + self.data_tail_size())
                                      * tp_size], 0.0), 1.0))
                        _thr = max(0.0, 2.0 * _f9 - 1.0)
                    tests.append({'stophere': False,
                                  'test': t['test'], 'data': t['data'],
                                  'threshold': _thr})
            elif t_i['type'] == 'buy':
                # THE TEST STARTS FROM THE FAMILY ASKED FOR (April 4, and
                # her ruling restoring it 2026-08-20). The buy has already
                # decoded its item above; its family is taken from that
                # item, and every test selector decodes from inside
                # test_<family>. A want for a vectorSpace therefore cannot
                # name a test of any other family - the family is where the
                # walk STARTS, not a label checked afterwards. This keeps
                # the want and the standard on separate floats, so a buyer
                # can refine how it judges without changing what it asks
                # for, and move what it asks for without losing its
                # standard.
                item_type = self.item_type(t_i['item'])
                _cat = 'test_' + str(item_type)
                seen_td = set()
                for pick in selectors:
                    if pick is None or not item_type:
                        continue
                    e_off = bank_offset + pick * self.bank_entry_size()
                    if self.p.get('single_float_ontology', True):
                        # a21: a test path and a data path are one float
                        # each, read from the category they hang under.
                        test_item = self.ontology_item_single(
                            float_vec[e_off], category=_cat)
                        data_item = self.ontology_item_single(
                            float_vec[e_off + tp_size], category='data')
                        # THE TAIL: corpora only, so a joining head always
                        # has operands it can actually take.
                        _tail = [
                            self.ontology_item_single(
                                float_vec[e_off + (2 + _j) * tp_size],
                                category='data', terminals_only=True)
                            for _j in range(self.data_tail_size())]
                    else:
                        test_item = self.ontology_item(
                            float_vec[e_off:e_off + tp_size],
                            category=_cat)
                        data_item = self.ontology_item(
                            float_vec[e_off + tp_size:e_off + 2 * tp_size],
                            category='data')
                        _tail = [
                            self.ontology_item(
                                float_vec[e_off + (2 + _j) * tp_size:
                                          e_off + (3 + _j) * tp_size],
                                category='data', terminals_only=True)
                            for _j in range(self.data_tail_size())]
                    if (test_item, data_item) in seen_td:
                        continue  # a conjunction with itself is itself
                    seen_td.add((test_item, data_item))
                    _thr = 0.0
                    if self.p.get('test_threshold'):
                        # HER MAP: bar = max(0, 2f-1) - ignorance
                        # demands nothing; standards are earned.
                        _f9 = float(min(max(
                            float_vec[e_off + (2 + self.data_tail_size())
                                      * tp_size], 0.0), 1.0))
                        _thr = max(0.0, 2.0 * _f9 - 1.0)
                    tests.append({'stophere': False,
                                  'test': test_item, 'data': data_item,
                                  'data_tail': list(_tail),
                                  'threshold': _thr})
            # The back-fill that set a buy's item FROM its test is gone with
            # ask_with_test_only (her ruling 2026-08-20). The item is
            # decoded in its own right above and the test starts from it,
            # so there is nothing to derive and nothing that could disagree
            # by construction rather than by being correct.
            # HER COBB-DOUGLAS SCENARIO (4 Oct 2026): "yes unpin"; "utility determines what they
            # ask for". A CD human's buy row asks for the good its own gene names - the text is chosen,
            # not pinned (her v14 corpus pin lifted for these humans only) - and the choice is selected
            # by the human's Cobb-Douglas utility, which is its fitness.
            if '_cd_choice' in t_i or '_cd_good_tree' in t_i:
                if '_cd_good_tree' in t_i:
                    _good = t_i.pop('_cd_good_tree')          # decision 18: the good read from the tree above
                    t_i.pop('_cd_choice', None)
                else:
                    _cx = min(max(float(t_i.pop('_cd_choice')), 0.0), 1.0 - 1e-12)
                    _goods = self.p['humans_cd']['goods']
                    _good = _goods[min(len(_goods) - 1, int(_cx * len(_goods)))]
                if t_i.get('type') == 'buy':
                    t_i['cd_good'] = _good
                    _spec = (self.p['humans_cd'].get('good_specs') or {}).get(_good)
                    if _spec:
                        # HER COBB-DOUGLAS HUMANS IN THE MACHINE-LEARNING WORLD (5 Oct 2026, "2 yes cd"): a good
                        # there differs in kind, test and corpus together, so the good the row's gene chooses sets
                        # the row's item and its tests and corpus from the configuration's good_specs. Worlds that
                        # list goods by name only (the cipher worlds) take the branch below, unchanged.
                        t_i['item'] = _spec['item']
                        for _t in tests:
                            if not _t.get('stophere'):
                                _t['test'] = _spec['test']
                                _t['data'] = _spec['data']
                    else:
                        for _t in tests:
                            if not _t.get('stophere'):
                                _t['data'] = 'data_cipher_' + _good
            tests.append({'stophere': True, 'test': 'stop', 'data': 'stop'})
            t_i['tests'] = tests

        return (trade_plan, float_vec)

    # trade_plan_to_float_vec was caller-less dead code; removed in the
    # 2026-08-10 genome rewrite (git history keeps it).

    def blank_message(self):
        cursor = 0
        trade_plan = OrderedDict([('type', self.__class__.__name__)])
        trade_plan['label'] = trade_plan['type'] + " Agent " + str(self.unique_id)

        # First find the sign, which is the raw float representation

        trade_plan['sign'] = [0.0]*self.p['sign_size']


        trade_plan['trades'] = []
        trade_plan['trades'].append(dict())
        trade_plan['trades'][0]['type'] = 'stop'

        return trade_plan

    def get_bought_items(self):
        bought_items = OrderedDict()
        for trade in self.message['trades']:
            if ('code' in trade) and ('pickle' in trade) and trade['code'] and trade['pickle']:
                bought_items[trade['pickle']] = trade['code']

        return bought_items