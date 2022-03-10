Each sequence carries a name that shorty describies its purpose. They are used to test the correct behaviour of the pediction/estimation system.
- 8\_shape: user walked an 8 shaped trajectory
- straight: is the test trajectory. This should be the one where measurements are performed
- straight\_reversed: is the same as straight but landmarks are crossed in the reverse order. A system should be able to successfully estimate both straigh and its reverse.
- weinberg\_sequences: 10 trajectory of: 5 forward (f) and 5 reverse (r). The user walked the same trajectory each time. The goals is to indentify the weinberg gain evolves based on : forward and reverse or some other causes.
- Ronin: Used Ronin's app and its neural network to predict the trajectory for comparison purpose.
- forward: this trajectory is the same the 3 other users to test system invariance.