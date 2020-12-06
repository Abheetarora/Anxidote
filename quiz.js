// select all elements
const start = document.getElementById("start");
const quiz = document.getElementById("quiz");
const question = document.getElementById("question");

const choiceA = document.getElementById("A")
const choiceB = document.getElementById("B");
const choiceC = document.getElementById("C");
const choiceD = document.getElementById("D");
const counter = document.getElementById("counter");

const progress = document.getElementById("progress");
const scoreDiv = document.getElementById("scoreContainer");

// create our questions
let questions = [
    {
        question : "I found it hard to wind down",
        choiceA : "Did not apply to me at all",
        choiceB : "Applied to me to some degree",
        choiceC : "Applied to me to more degree",
        choiceD : "Applied to me most of the time"
        
    },{
        question : "I was aware of dryness of my mouth",
        choiceA : "Did not apply to me at all",
        choiceB : "Applied to me to some degree",
        choiceC : "Applied to me to more degree",
        choiceD : "Applied to me most of the time"
    },{
        question : "I couldn’t seem to experience any positive feeling at all",
        choiceA : "Did not apply to me at all",
        choiceB : "Applied to me to some degree",
        choiceC : "Applied to me to more degree",
        choiceD : "Applied to me most of the time"
    },{
        question : "I experienced breathing difficulty (e.g. excessively rapid breathing, breathlessness in the absence of physical exertion)",
        choiceA : "Did not apply to me at all",
        choiceB : "Applied to me to some degree",
        choiceC : "Applied to me to more degree",
        choiceD : "Applied to me most of the time"
    },{
        question : "I found it difficult to work up the initiative to do things",
        choiceA : "Did not apply to me at all",
        choiceB : "Applied to me to some degree",
        choiceC : "Applied to me to more degree",
        choiceD : "Applied to me most of the time"
    },{
        question : "I tended to over-react to situations",
        choiceA : "Did not apply to me at all",
        choiceB : "Applied to me to some degree",
        choiceC : "Applied to me to more degree",
        choiceD : "Applied to me most of the time"
    },{
        question : "I experienced trembling (e.g. in the hands)",
        choiceA : "Did not apply to me at all",
        choiceB : "Applied to me to some degree",
        choiceC : "Applied to me to more degree",
        choiceD : "Applied to me most of the time"
    },{
        question : "I felt that I was using a lot of nervous energy",
        choiceA : "Did not apply to me at all",
        choiceB : "Applied to me to some degree",
        choiceC : "Applied to me to more degree",
        choiceD : "Applied to me most of the time"
    },{
        question : "I was worried about situations in which I might panic and make a fool of myself",
        choiceA : "Did not apply to me at all",
        choiceB : "Applied to me to some degree",
        choiceC : "Applied to me to more degree",
        choiceD : "Applied to me most of the time"
    },{
        question : "I felt that I had nothing to look forward to",
        choiceA : "Did not apply to me at all",
        choiceB : "Applied to me to some degree",
        choiceC : "Applied to me to more degree",
        choiceD : "Applied to me most of the time"
    },{
        question : "I found myself getting agitated",
        choiceA : "Did not apply to me at all",
        choiceB : "Applied to me to some degree",
        choiceC : "Applied to me to more degree",
        choiceD : "Applied to me most of the time"
    },{
        question : "I found it difficult to relax",
        choiceA : "Did not apply to me at all",
        choiceB : "Applied to me to some degree",
        choiceC : "Applied to me to more degree",
        choiceD : "Applied to me most of the time"
    },{
        question : "I felt down-hearted and blue ",
        choiceA : "Did not apply to me at all",
        choiceB : "Applied to me to some degree",
        choiceC : "Applied to me to more degree",
        choiceD : "Applied to me most of the time"
    },{
        question : "I was intolerant of anything that kept me from getting on with what I was doing",
        choiceA : "Did not apply to me at all",
        choiceB : "Applied to me to some degree",
        choiceC : "Applied to me to more degree",
        choiceD : "Applied to me most of the time"
    },{
        question : "I felt I was close to panic",
        choiceA : "Did not apply to me at all",
        choiceB : "Applied to me to some degree",
        choiceC : "Applied to me to more degree",
        choiceD : "Applied to me most of the time"
    },{
        question : "I was unable to become enthusiastic about anything",
        choiceA : "Did not apply to me at all",
        choiceB : "Applied to me to some degree",
        choiceC : "Applied to me to more degree",
        choiceD : "Applied to me most of the time"
    },{
        question : "I felt I wasn’t worth much as a person ",
        choiceA : "Did not apply to me at all",
        choiceB : "Applied to me to some degree",
        choiceC : "Applied to me to more degree",
        choiceD : "Applied to me most of the time"
    },{
        question : "I felt that I was rather touchy ",
        choiceA : "Did not apply to me at all",
        choiceB : "Applied to me to some degree",
        choiceC : "Applied to me to more degree",
        choiceD : "Applied to me most of the time"
    },{
        question : "I was aware of the action of my heart in the absence of physical exertion (e.g. sense of heart rate increase, heart missing a beat)",
        choiceA : "Did not apply to me at all",
        choiceB : "Applied to me to some degree",
        choiceC : "Applied to me to more degree",
        choiceD : "Applied to me most of the time"
    },{
        question : "I felt scared without any good reason",
        choiceA : "Did not apply to me at all",
        choiceB : "Applied to me to some degree",
        choiceC : "Applied to me to more degree",
        choiceD : "Applied to me most of the time"
    },{
        question : "I felt that life was meaningless",
        choiceA : "Did not apply to me at all",
        choiceB : "Applied to me to some degree",
        choiceC : "Applied to me to more degree",
        choiceD : "Applied to me most of the time"
    }
];

// create some variables

const lastQuestion = questions.length - 1;
let runningQuestion = 0;
let count = 0;

let score = 0;

// render a question
function renderQuestion(){
    let q = questions[runningQuestion];
    
    question.innerHTML = "<p>"+ q.question +"</p>";
    choiceA.innerHTML = q.choiceA;
    choiceB.innerHTML = q.choiceB;
    choiceC.innerHTML = q.choiceC;
    choiceD.innerHTML = q.choiceD;
}

start.addEventListener("click",startQuiz);

// start quiz
function startQuiz(){
    start.style.display = "none";
        renderQuestion();
        quiz.style.display = "block";
       
    //}    
    
    
}

// render progress
//function renderProgress(){
    //for(let qIndex = 0; qIndex <= lastQuestion; qIndex++){
       // progress.innerHTML += "<div id="+ runningQuestion +"></div>";
    //}
//}

// counter render

/* function renderCounter(){
    if(count <= questionTime){
        counter.innerHTML = count;
        timeGauge.style.width = count * gaugeUnit + "px";
        count++
    }else{
        count = 0;
        // change progress color to red
        answerIsWrong();
        if(runningQuestion < lastQuestion){
            runningQuestion++;
            renderQuestion();
        }else{
            // end the quiz and show the score
            clearInterval(TIMER);
            scoreRender();
        }
    } 
} */

// checkAnwer

function checkAnswer(answer){
    if( answer == questions[runningQuestion].choiceB){
        score++;
    }else if( answer == questions[runningQuestion].choiceC){
        score += 2;
    }else if( answer == questions[runningQuestion].choiceD){
        score += 3;
    } 
    runningQuestion++;
    if(runningQuestion <= lastQuestion) {
        renderQuestion();
    } else {
        scoreRender();
    }
} 


// score render
function scoreRender(){
    scoreDiv.style.display = "block";
    
    // calculate the amount of question percent answered by the user
    //const scorePerCent = Math.round(100 * score/questions.length);
    
    
    scoreDiv.innerHTML += "<p>"+ score +"</p>";
}





















