import random

print("Rules of the Rock Paper Scissor game as follows: \n"
                                +"Rock vs Paper->Paper wins \n"
                                + "Paper vs Scissor->Scissor wins \n"
                                +"Scissor vs Rock->Rock wins \n")

user_action = input("Enter some action like ( Rock, Paper, Scissors ) : \n ")

actions = ['Rock', 'Paper', 'Scissors']
comp_action = random.choice(actions)

print("\n You Chose : ", user_action, "\n Computer Chose : ", comp_action)

if user_action == comp_action:
    print("You Both Selected ", user_action, " It's a Tie.")

elif user_action == "Rock":
    if comp_action == "Scissors":
        print("Rock smashes Scissors, You Win, Computer Lose.")

    else:
        print("Paper Covers Rock, Computer Wins, You Lose.")

elif user_action == "Scissors":
    if comp_action == "Paper":
        print("Scissors cut Paper, You Win, Computer Lose.")

    else:
        print("Rock Smashes Scissors, Computer Wins, You Lose.")

elif user_action == "Paper":
    if comp_action == "Rock":
        print("Paper Covers Rock, You Win, Computer Lose.")

    else:
        print("Scissors Cut Paper, Computer Wins, You Lose.")
