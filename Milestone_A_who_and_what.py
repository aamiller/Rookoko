#!/usr/bin/python3
'''Milestone_A_who_and_what.py
This runnable file will provide a representation of
answers to key questions about your project in CSE 415.

'''

# DO NOT EDIT THE BOILERPLATE PART OF THIS FILE HERE:

CATEGORIES=['Baroque Chess Agent','Feature-Based Reinforcement Learning for the Rubik Cube Puzzle',\
  'Supervised Learning: Comparing Trainable Classifiers']

class Partner():
  def __init__(self, lastname, firstname, uwnetid):
    self.uwnetid=uwnetid
    self.lastname=lastname
    self.firstname=firstname

  def __lt__(self, other):
    return (self.lastname+","+self.firstname).__lt__(other.lastname+","+other.firstname)

  def __str__(self):
    return self.lastname+", "+self.firstname+" ("+self.uwnetid+")"

class Who_and_what():
  def __init__(self, team, option, title, approach, workload_distribution, references):
    self.team=team
    self.option=option
    self.title=title
    self.approach = approach
    self.workload_distribution = workload_distribution
    self.references = references

  def report(self):
    rpt = 80*"#"+"\n"
    rpt += '''The Who and What for This Submission

Final Project in CSE 415, University of Washington, Winter, 2018
Milestone A

Team: 
'''
    team_sorted = sorted(self.team)
    # Note that the partner whose name comes first alphabetically
    # must do the turn-in.
    # The other partner(s) should NOT turn anything in.
    rpt += "    "+ str(team_sorted[0])+" (the partner who must turn in all files in Catalyst)\n"
    for p in team_sorted[1:]:
      rpt += "    "+str(p) + " (partner who should NOT turn anything in)\n\n"

    rpt += "Option: "+str(self.option)+"\n\n"
    rpt += "Title: "+self.title + "\n\n"
    rpt += "Approach: "+self.approach + "\n\n"
    rpt += "Workload Distribution: "+self.workload_distribution+"\n\n"
    rpt += "References: \n"
    for i in range(len(self.references)):
      rpt += "  Ref. "+str(i+1)+": "+self.references[i] + "\n"

    rpt += "\n\nThe information here indicates that the following file will need\n"+\
     "to be submitted (in addition to code and possible data files):\n"
    rpt += "    "+\
     {'1':"Baroque_Chess_Agent_Report",'2':"Rubik_Cube_Solver_Report",\
      '3':"Trainable_Classifiers_Report"}\
        [self.option]+".pdf\n"

    rpt += "\n"+80*"#"+"\n"
    return rpt

# END OF BOILERPLATE.

# Change the following to represent your own information:

adele = Partner("Adele", "Miller", "adelem")
cillian = Partner("Cillian", "Murphy", "camurphy")
team = [adele, cillian]

OPTION = '1'
# Legal options are 1, 2, and 3.

title = "Rookoko: An Exuberant Chess Player"
 # In this case, the Python file for the formulation would be named End_Poverty.py.

approach = '''Our approach will be to use the minimax algorithm in addition to
iterative deepening, Zobrist hashing, and alpha-beta pruning. Our static evaluation
will first need to know what the relative values of each piece is, so that we can avoid
getting valuable pieces captured and try to get our opponent's valuable pieces. 
Firstly, we will need to code our move generator, looking at each individual piece and
then applying the movement rules to those pieces. We intend to pre-calculate the first 
few state values following early moves in the game.'''

workload_distribution = '''Cillian will have the primary responsibilty for the static eval
function and making sure the code is efficient. He will also have the responsibility of
making puns for our agent to say to the other agent, for his is an unrepenting pun-master.
Adele will have the primary responsibility for implementing alpha-beta pruning and Zobrist hashing,
with some work on the move-generation code.'''

reference1 = '''Wikipedia article on Baroque Chess;
    URL: https://en.wikipedia.org/wiki/Baroque_chess (accessed Feb. 25, 2018)'''

reference2 = '''Ultima" by Hans L. Bodlaender,
    available online at: http://www.chessvariants.com/other.dir/ultima.html'''

our_submission = Who_and_what([adele, cillian], OPTION, title, approach, workload_distribution, [reference1, reference2])

# You can run this file from the command line by typing:
# python3 who_and_what.py

# Running this file by itself should produce a report that seems correct to you.
if __name__ == '__main__':
  print(our_submission.report())
