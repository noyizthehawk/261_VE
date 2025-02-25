"""
Solution stub for implementation of variable elimination.

1. Complete the implementation of the `Factor` class to implement missing functionality.
2. Implement the variable elimination algorithm in the `var_elimination` function
3. Implement `example_bn` to return a collection of factors representing the
   belief network in the last question of the assignment.  (It's okay to
   hardcode the creation of `Factor` objects in `example_bn`.)

We will test your solution by calling ``python3 ve.py`` from the shell prompt.
You may add additional code to the end of the `main` function if you desire,
but do not remove or modify the calls at the beginning.  You may add additional
tests to the `unit_tests` function if you desire.
"""
import numpy as np
import itertools


def var_elimination(factors, observations, query_name, var_order):
    """
    Performs the variable elimination to compute the distribution of
    `query_name` conditional on `observations`.

    Args:
      factors:
        Collection of `Factor` objects representing the conditional
        distributions of the random variables in the belief network.
      observations:
        Dictionary where the keys are variable names and the values are the
        observed value for the named variable.
      query_name:
        Name of the variable to compute a posterior distribution factor for.
      var_order:
        Sequence of variable names indicating the variable ordering to use.

    Returns:
      A `Factor` object with a single variable (`query_name`), with the results
      of the query.
    """
    #TODO
    

def example_bn():
    """
    Returns a collection of `Factor` objects representing the belief network
    in the assignment.
    """
    return [
        # Pr(A)
        Factor({'A': (0,1)},
               ((0, 0.25),
                (1, 0.75))),

        # TODO
        ]

class Factor(object):
    """Represents a factor with associated operations for variable elimination."""
    
    def __init__(self, domains, values=None):
        """
        Args:
          domains:
            A dictionary where the keys are variable names in the factor, and
            the values are tuples containing all possible value of the variable.
          values:
            Convenience argument for initializing the Factor's table.
            List of tuples, where each tuple is a row of the table.
            First element of each tuple is the value of the first variable, etc.
            Final element of each tuple is the value of the factor for the given
            combination of values. See `unit_tests` for example usage.
        """
                   
        self.domains = dict(domains)
        shape = [len(domains[v]) for v in domains]
        self.data = np.zeros(shape)

        if values is not None:
            for v in values:
                key = v[:-1]
                val = v[-1]
                self[key] = val

    # ------- Operators
    def condition(self, name, val):
        """Return a new factor that conditions on ``name=val``"""
        j = tuple(self.names).index(name)
        new_domains = dict(self.domains) # copy own domains...
        del new_domains[name]            # ... except for `name`
        new_f = Factor(new_domains)
        # TODO
        return new_f

    def sum_out(self, name):
        """Return a new factor that sums out variable `name`"""
        j = tuple(self.names).index(name)
        new_domains = dict(self.domains) # copy own domains...
        del new_domains[name]            # ... except for `name`
        new_f = Factor(new_domains)
        # TODO
        return new_f

    def normalize(self):
        """Return a new factor whose values add to 1"""
        new_f = Factor(self.domains)
        new_f.data = self.data / np.sum(self.data)
        return new_f

    def __mul__(self, other):
        """Construct a new factor by multiplying `self` by `other`"""
        # Figure out the variables and domains for the new factor
        new_domains = dict(self.domains)
        for name,domain in other.domains.items():
            if name not in new_domains:
                new_domains[name] = domain
            elif self.domain(name) != other.domain(name):
                raise ValueError(f"Incompatible domains for {repr(name)}: "
                                 f"{repr(self.domain(name))} versus "
                                 f"{repr(other.domain(name))}")

        # Empty factor with the computed domains
        new_f = Factor(new_domains)

        # Perform the multiplications
        for k in new_f.keys:
            h = dict(zip(new_f.names, k))
            k1 = tuple(h[name] for name in self.names)
            k2 = tuple(h[name] for name in other.names)
            new_f[k] = self[k1] * other[k2]

        return new_f

    # ------- Accessors
    @property
    def names(self):
        """Return the names of all the variable in the table"""
        return tuple(self.domains.keys())

    @property
    def keys(self):
        """Iterate over all value combinations for all variables in table"""
        return tuple(itertools.product(*self.domains.values()))

    @property
    def size(self):
        return self.data.size

    def domain(self, name):
        """Return the domain of values for variable `name`"""
        return tuple(self.domains[name])

    def __getitem__(self, key):
        """Return the table entry for the tuple of values `key`"""
        if type(key) is not tuple:
            key = (key,)
        if len(key) != len(self.names):
            raise ValueError(f"Wrong number of arguments:"
                             f"{len(key)} instead of {len(self.names)}")
        idx = tuple(self._idx(name,val) for (name,val) in zip(self.names, key))
        return self.data[idx]

    def __setitem__(self, key, new_val):
        """Set the table entry for the tuple of values `key` to `new_val`"""
        if len(key) != len(self.names):
            raise ValueError(f"Wrong number of arguments: "
                             f"{len(key)} instead of {len(self.names)}")
        idx = tuple(self._idx(name,val) for (name,val) in zip(self.names, key))
        self.data[idx] = new_val

    def _idx(self, name, val):
        """Return the index of `val` in `name`s domain"""
        try:
            return self.domains[name].index(val)
        except ValueError:
            raise ValueError(f"{repr(val)} is not in domain of {repr(name)}")

    # ------- Standard overrides for pretty printing
    def __repr__(self):
        cls = self.__class__.__name__
        return f"<{cls} object: names={list(self.names)}, rows={self.size}>"

    def __str__(self):
        w = 0
        for k in self.keys:
            for v in k:
                w = max(w, len(str(v)))
        fmt = f"%{w}s  " * len(self.names)
        out = fmt % tuple(self.names) + "value\n"
        out += fmt % tuple("-"*w for n in self.names) + "-----"
        for k in self.keys:
            out += "\n"
            out += fmt % k
            out += f"{self[k]}"
        return out

def unit_tests():
    f = Factor({'x':(1,2,3), 'y':(0,1)},
               [(1,0,0.5),
                (1,1,0.6),
                (2,0,0.1),
                (2,1,0.2),
                (3,0,0.8),
                (3,1,1.0)])
    g = Factor({'y':(0,1), 'z':('a','b')})
    g[0,'b'] = 9
    
    fg = f*g
    assert fg.names == ('x', 'y', 'z')
    assert np.isclose(fg[1,0,'a'], 0.0)
    assert np.isclose(fg[1,0,'b'], 4.5)
    assert np.isclose(fg[3,0,'b'], 7.2)

    try:
        z = Factor({'y':(1,0), 'z':('a','b')})
        fz = f*z
        assert False, "expected error"
    except:
        pass
    
    h = f.condition('x', 3)
    assert len(h.keys) == 2
    assert h.names == ('y',)
    assert np.isclose(h[0], 0.8)
    assert np.isclose(h[1], 1.0)

    z = h.normalize()
    assert np.isclose(z[0], 0.8/1.8)
    assert np.isclose(z[1], 1.0/1.8)

    h = f.sum_out('y')
    assert len(h.keys) == 3
    assert h.names == ('x',)
    assert np.isclose(h[1], 1.1)
    assert np.isclose(h[2], 0.3), h[2]
    assert np.isclose(h[3], 1.8)
    
    ### Simpson's paradox
    # joint distribution
    joint = Factor({'A':('y','o'), 'D':(True, False), 'R':(True, False)},
                   [('y', True,  True,  0.225),
                    ('y', True,  False, 0.15),
                    ('y', False, True,  0.0875),
                    ('y', False, False, 0.0375),
                    ('o', True,  True,  0.025),
                    ('o', True,  False, 0.1),
                    ('o', False, True,  0.1125),
                    ('o', False, False, 0.2625)])
    # Pr(A) computed manually from joint
    A = Factor({'A':('y','o')},
               [('y', 0.5),
                ('o', 0.5)])
    # Pr(D|A) computed manually from joint
    D = Factor({'D':(True,  False), 'A':('y', 'o')},
               [(True,  'y', 0.75),
                (False, 'y', 0.25),
                (True,  'o', 0.25),
                (False, 'o', 0.75)])
    # Pr(R|D,A)
    R = Factor({'R':(True, False), 'D':(True, False), 'A':('y', 'o')},
               [(True,  True,  'y', 0.60),
                (True,  False, 'y', 0.70),
                (False, True,  'y', 0.40),
                (False, False, 'y', 0.30),
                (True,  True,   'o', 0.20),
                (True,  False, 'o', 0.30),
                (False, True,  'o', 0.80),
                (False, False, 'o', 0.70)])

    ad1 = A * D                 # P(A) * P(D|A)
    ad2 = joint.sum_out('R')    # sum_R P(A,D,R)
    for k in ad1.keys:
        assert np.isclose(ad1[k], ad2[k])

    joint2 = A * D * R
    for k in joint.keys:
        assert np.isclose(joint[k], joint2[k])

    # Observational probabilities from slide
    rd = joint.sum_out('A')
    r_dt = rd.condition('D', True).normalize()   # Pr(R | D=true)
    assert np.isclose(r_dt[True], 0.50)
    r_df = rd.condition('D', False).normalize()  # Pr(R | D=false)
    assert np.isclose(r_df[True], 0.40)

    # if we got here, everything passed
    return 'ok'

def main():
    unit_tests()
    factors = example_bn()
    solution = var_elimination(factors, {'G':0, 'E':1}, 'B',
                               ['G','E','A','B','C','D','F'])

    print(f"P(B=1 | G=0, E=1) = {solution[1]}")
    print(f"P(B=0 | G=0, E=1) = {solution[0]}")

if __name__ == '__main__':
    main()
