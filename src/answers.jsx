const ans=[
    {
        id:1,
        name:"Comments",
        code:
        `
        #include<stdio.h>
        #include<string.h>
        void main()
        {
        int len,i,e,cs,count,count1;
        char str[100],ch[4];
        enum states{
        q0,q1,q2,q3,q4
        };
        do
        {
        i=0;count=0;count1=0;cs=q0;
        printf("enter the data: ");
        fflush(stdin);
        gets(str);
        len=strlen(str);
        while(i<=len)
        {
        switch(cs)
        {
        case q0:if(str[i]=='/')
        cs=q1;
        else
        e=0;
        break;
        case q1:if(str[i]=='/')
        cs=q2;
        else if(str[i]=='*')
        {
        count++;
        cs=q3;
        }
        else
        {
        e=0;
        i=len;
        }
        break;
        case q2:if((str[i]=='\n')||i<=60)
        {
        if(i==len)
        e=1;
        else
        cs=q2;
        }
        else
        e=0;
        break;
        case q3:if((str[i]=='*') && (str[i+1]=='/'))
        {
        count1++;
        if(str[i+2]=='\0')
        {
        cs=q4;
        e=1;
        i=i+2;
        }
        else
        cs=q2;
        }
        else
        e=0;
        break;
        case q4:e=1;
        break;
        default:break;
        }
        i++;
        }
        if(e==1&&count==count1)
        printf("\nenterd data is comment \n");
        else
        printf("\nenterd data is not comment \n");
        printf("\ndo u want to continue to enter another string:(y/n): ");
        scanf("%s",&ch);
        }while(ch[0]=='y');
        }


       -----------------------------------------------------------------------------------------------------------------
        Output:
        /* enter the data: //hello welcome to cd
        enterd data is comment
        do u want to continue to enter another string:(y/n): y
        enter the data: /*hi how are all?*/
        /*enterd data is comment
        do u want to continue to enter another string:(y/n): y
        enter the data: /good to see you
        enterd data is not comment
        do u want to continue to enter another string:(y/n): y
        enter the data: /* bye lets meet tommorow
        enterd data is not comment
        do u want to continue to enter another string:(y/n): n
        -----------------------------------------------------------------------------------------------------------------
`
    },
    {
        id:2,
        name:"Keywords",
        code:
        `
        #include<stdio.h>
        #include<string.h>
        void main(){
        char input[10],com[10],cs,ch[4];
        int size,i=0,e,len,j,k=0;
        enum states{q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,q16,qf};
        do{
        i=0;    
        k=0;
        printf("\nenter the input:");
        fflush(stdin);
        scanf("%s",input);
        cs=q0;
        size=strlen(input);
        for(len=0;len<=3;len++){
        switch(cs){
        case q0:if(input[i]=='a')
        cs=q1;if(input[i]=='b')
                         cs=q2;
        if(input[i]=='c')
        cs=q3
        if(input[i]=='d')
        cs=q4;
        if(input[i]=='e')
        cs=q5;
        if(input[i]=='f')
        cs=q6;
        if(input[i]=='g')
        cs=q7;
        if(input[i]=='i')
        cs=q8;
        if(input[i]=='l')
        cs=q9;
        if(input[i]=='r')
        cs=q10;
        if(input[i]=='s')
        cs=q11;
        if(input[i]=='t')
        cs=q12;
        if(input[i]=='u')
        cs=q13;
        if(input[i]=='v')
        cs=q14;
        if(input[i]=='w')
        cs=q15;
        i++;
        for(j=1;j<=size;j++){
        com[k]=input[j];
        k++;
        }
        break;
        case q1:if(strcmp(com,"uto")==0)
        cs=qf;
        else
        break;
        case q2:if(strcmp(com,"reak")==0)
        cs=qf;
        else if(strcmp(com,"oolean")==0)
        cs=qf;
        else
        break;
        case q3:if(strcmp(com,"har")==0||strcmp(com,"ase")==0||strcmp(com,"ontinue")==0||strcmp(com,"onst")==0)
        cs=qf;
        else
        break;
        case q4:if(strcmp(com,"ouble")==0)
        cs=qf;
        else if(strcmp(com,"efault")==0)
        cs=qf;
        else if(strcmp(com,"o")==0)
        cs=qf;
        else
        break;
        case q5:if(strcmp(com,"num")==0)
        cs=qf;
        else if(strcmp(com,"lse")==0)
        cs=qf;
        else if(strcmp(com,"xtern")==0)
        cs=qf;
        else
        break;
        case q6:if(strcmp(com,"or")==0)
        cs=qf;
        else if(strcmp(com,"loat")==0)
        cs=qf;
        else
        break;
        case q7:if(strcmp(com,"oto")==0)
        cs=qf;
        else
        break;
         case q8:if(strcmp(com,"f")==0)
        cs=qf;
        else if(strcmp(com,"nt")==0)
        cs=qf;
        else
        break;
        case q9:if(strcmp(com,"ong")==0)
        cs=qf;
        else
        break;
        case q10:if(strcmp(com,"egister")==0)
        cs=qf;
        else if(strcmp(com,"eturn")==0)
        cs=qf;
        else
        break;
        case q11:if(strcmp(com,"truct")==0||
        strcmp(com,"witch")==0||
        strcmp(com,"izeof")==0||
        strcmp(com,"hort")==0||
        strcmp(com,"igned")==0||
        strcmp(com,"tatic")==0)
        cs=qf;
        else
        break;
        case q12:if(strcmp(com,"ypedef")==0)
        cs=qf;
        else
        break;
        case q13:if(strcmp(com,"nion")==0||strcmp(com,"nsigned")==0)
        cs=qf;
        else
        break;
        case q14:if(strcmp(com,"oid")==0||strcmp(com,"olatile")==0)
        cs=qf;
        else
        break;
        case q15:if(strcmp(com,"hile")==0)
        cs=qf;
        else
        break;
        case qf:break;
        default:break;
        }   
        }
        if(cs==qf)
        printf("\na valid keyword");
        else
        printf("\nnot a valid keyword");
        printf("\nDo u want to continue(y/n):");
        scanf("%s",&ch);
        }while(ch[0]=='y');
        }


        -----------------------------------------------------------------------------------------------------------------

        Output:
        /*
        enter the input:auto
        a valid keyword
        Do u want to continue(y/n):y
        enter the input:const
        a valid keyword
        Do u want to continue(y/n):y
        enter the input:while
        a valid keyword
        Do u want to continue(y/n):y
        enter the input:what
        not a valid keyword
        Do u want to continue(y/n):y
        enter the input:goto
        a valid keyword
        Do u want to continue(y/n):n
        -----------------------------------------------------------------------------------------------------------------
        `
    },
    {
        id:3,
        name:"Validating Identifiers",
        code:
        `
        #include<stdio.h>
        #include<conio.h>
        #include<string.h>
        void main()
        {
        intlen,i,cstate,error=0;
        charstr[10],ch[4];
        enum states{q0,q1};
        do
        {
        i=0;cstate=q0;
        printf("\n enter the string:");
        fflush(stdin);
        gets(str);
        len=strlen(str);
        if(len<=8)
        {
         while(i<len)
         {
         switch(cstate)
         {
         case q0: if((str[i]>='a'&&str[i]<='z')||(str[i]>='A'&&str[i]<='Z')||(str[i]=='_'))
        cstate=q1;
         else
        error=1;
         i++;
         break;
        caseq1: if( !((str[i]>='a' && str[i]<='z') || (str[i]>='A' && str[i]<='Z') || (str[i]>='0' &&
        str[i]<='9') || (str[i]=='_')))
        error=1;
        i++;
        break;
         }
        }
        if(error==1)
        printf("invalid identifier");
        else
        printf("it is a valid identifier");
        }
        else
        printf("invalid identifiers");
        printf("\n do u want to continue to enter another string:(y/n):");
        scanf("%s",&ch);
        }while(ch[0]=='y');
        getch();
        }


        -----------------------------------------------------------------------------------------------------------------
        Output:
        enter the string:cse
        it is a valid identifier
        do u want to continue to enter another string:(y/n):y
        enter the string:a1
        it is a valid identifier
        do u want to continue to enter another string:(y/n):y
        enter the string:1a
        invalid identifier
        do u want to continue to enter another string:(y/n):n
        -----------------------------------------------------------------------------------------------------------------
        `
    },
    {
        id:4,
        name:"Operators",
        code:
        `
#include<stdio.h>
#include<conio.h>
#include<string.h>
void main(){
inti,error=0,s,len;
char input[20],ch;
typedef enum states{q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,qf}states;
states curstate;
do{
error=0;
s=0;
curstate=q0;
printf("enter the input:");
fflush(stdin);
gets(input);
len=strlen(input);
for(i=0;i<=len;i++){
if(i<=2){
switch(curstate){
case q0:if(input[i]=='!'||input[i]=='='){
if(len==1)
curstate=q6;
else
curstate=q1;
}
else if (input[i]=='>'){
if(len==1)
curstate=q6;
else
curstate=q8;
}
else if(input[i]=='<'){
if(len==1)
curstate=q6;
else
curstate=q9;
}
else if(input[i]=='+'){
if(len==1)
curstate=q6;
else
curstate=q2;
}
else if(input[i]=='?'){
if(len==1)
error=1;
else
curstate=q7;
}
else if(input[i]=='-'){
if(len==1)
curstate=q6;
else
curstate=q3;
}
else if(input[i]=='*'||input[i]=='^'||input[i]==',')
curstate=q6;
else if(input[i]=='/'||input[i]=='^'||input[i]=='%')
curstate=q6;
else if(input[i]='&'){
if(len==1)
error=1;
else
curstate=q4;
}
else if(input[i]=='!'){
if(len==1)
curstate=q6;
else
curstate=q5;
}
else
error=1;
break;
case q1:if(input[i]=='=')
curstate=qf;
else
error=1;
break;
case q2:
if(input[i]=='+'||input[i]=='=')
curstate=qf;
else
error=1;
break;
case q3:if(input[i]=='-'||input[i]=='=')
curstate=qf;
else
error=1;
break;
case q4:if(input[i]=='&')
curstate=qf;
else
error=1;
break;
case q5:if(input[i]=='!')
curstate=qf;
else
error=1;
break;
case q6:s=1;
error=0;
break;
case q7:if(input[i]==':')
curstate=qf;
else
error=1;
break;
case q8:
if(input[i]=='>'||input[i]=='=')
curstate=qf;
else
error=1;
break;
case q9:
if(input[i]=='<'||input[i]=='=')
curstate=qf;
else
error=1;
break;
caseqf:s=0;
error=0;
break;
}
}
}
if(strcmp(input,"sizeof")==0){
error=0;
len=2;
s=0;
}
if((error==0)&&(len==1)&&(s==1))
printf("%s is an operator",input);
else if(error==0&&len==2&&s==0)
printf("%s is an operator",input);
else
printf("%s is not an operator",input);
printf("\n do u want to continue(y/n):");
ch=getchar();
}while(ch!='n');
getch();
}


-----------------------------------------------------------------------------------------------------------------
output:
enter the input:+
+ is an operator
do u want to continue(y/n):y
enter the input:<
< is an operator
do u want to continue(y/n):y
enter the input:+=
+= is an operator
do u want to continue(y/n):y
enter the input:!
! is an operator
do u want to continue(y/n):y
enter the input:?:
?: is an operator
do u want to continue(y/n):n
-----------------------------------------------------------------------------------------------------------------
        `
    },
    {
        id:5,
        name:"DFA",
        code:
        `
#include <stdio.h>
#define TOTAL_STATES 2
#define FINAL_STATES 1
#define ALPHABET_CHARCTERS 2
#define UNKNOWN_SYMBOL_ERR 0
#define NOT_REACHED_FINAL_STATE 1
#define REACHED_FINAL_STATE 2
enum DFA_STATES{q0,q1};
enum input{a,b};
int Accepted_states[FINAL_STATES]={q1};
char alphabet[ALPHABET_CHARCTERS]={'a','b'};
int Transition_Table[TOTAL_STATES][ALPHABET_CHARCTERS];
int Current_state=q0;
void DefineDFA(){
Transition_Table[q0][a] = q1;
Transition_Table[q0][b] = q0;
Transition_Table[q1][a] = q1;
Transition_Table[q1][b] = q0;
}
int DFA(char current_symbol){
int i,pos;
for(pos=0;pos<ALPHABET_CHARCTERS; pos++)
if(current_symbol==alphabet[pos])
break;//stops if any character other than a or b
if(pos==ALPHABET_CHARCTERS)
return UNKNOWN_SYMBOL_ERR;
for(i=0;i<FINAL_STATES;i++)
if((Current_state=Transition_Table[Current_state][pos])
==Accepted_states[i])
return REACHED_FINAL_STATE;
return NOT_REACHED_FINAL_STATE;
}
void main(void){
char current_symbol;
int result;
DefineDFA(); //Fill transition table
printf("Enter a string with 'a' s and 'b's:\n Press Enter Key to stop\n");
while((current_symbol=getchar())!= '\n')
if((result= DFA(current_symbol))==UNKNOWN_SYMBOL_ERR)
break;
switch (result) {
case UNKNOWN_SYMBOL_ERR:printf("Unknown Symbol %c",
current_symbol);
break;
case NOT_REACHED_FINAL_STATE:printf("Not accepted"); break;
case REACHED_FINAL_STATE:printf("Accepted");break;
default: printf("Unknown Error");
}
printf("\n\n\n");
getch();
}


-----------------------------------------------------------------------------------------------------------------

/* output
Enter a string with 'a' s and 'b's:
Press Enter Key to stop
aaabbaaaaaaaaa
Accepted
Enter a string with 'a' s and 'b's:
Press Enter Key to stop
bbbbbbbba
Accepted
Enter a string with 'a' s and 'b's:
Press Enter Key to stop
bab
Not accepted
Enter a string with 'a' s and 'b's:
Press Enter Key to stop
abc
Unknown Symbol c
Enter a string with 'a' s and 'b's:
Press Enter Key to stop
aaabbaaaaaaaaa
Accepted
Enter a string with 'a' s and 'b's:
Press Enter Key to stop
bbbbbbbba
Accepted
Enter a string with 'a' s and 'b's:
Press Enter Key to stop
bab
Not accepted
Enter a string with 'a' s and 'b's:
Press Enter Key to stop
abc
Unknown Symbol c
*/

-----------------------------------------------------------------------------------------------------------------
        `
    },

    {
        id:6,
        name:"Constants",
        code:
        `
        #include<stdio.h>
#include<conio.h>
#include<string.h>
void main()
{
char in[50],x;
int i=0,e,l;
typedef enum states{q0,q1,q2,q3,q4}states;
states cs;
clrscr();
go:
e=0;
cs=q0;
fflush(stdin);
printf("\nEnter the constants:");
gets(in);
l=strlen(in);
for(i=0;i<l;i++)
{
switch(cs)
{
case q0:if((in[i]=='+'||in[i]=='-')||(in[i]>='0'&&in[i]<='9'))
	cs=q1;
       //else if(in[i]=='.')
	//cs=q2;
	else if((in[i]=='e'||in[i]=='E'))
	cs=q2;
	else
	e=1;
	break;
case q1:if(in[i]>='0'&&in[i]<='9')
	cs=q1;
	else if(in[i]=='.')
	cs=q2;
	else if((in[i]=='e'||in[i]=='E')&&(in[i+1]!='\0'))
	cs=q3;
	else
	e=1;
	break;
case q2:if(in[i]>='0'&&in[i]<='9')
	cs=q2;
	else if((in[i]=='e'||in[i]=='E')&&(in[i+1]!='\0'))
	cs=q4;
	else
	e=1;
	break;
case q3:if(in[i]=='+'||in[i]=='-')
	cs=q2;
	else
	e=1;
	break;
case q4:if((in[i]=='+'||in[i]=='-')||(in[i]>='0'&&in[i]<='9'))
	cs=q4;
	else
	e=1;
	break;
}
}
if(e==0)
printf("Given string is valid constant");
else
printf("Given string is not a constant");
printf("\n Do u wnat to continue(y/n):");
scanf("%s",&x);
if((x=='y')||(x=='Y'))
goto go;
getch();
}
        `
    },
    {
        id:7,
        name:"Conversion from NFA to DFA",
        code:
        `
#include<stdio.h> 
#include<stdlib.h> 
struct node { 
int st; 
struct node *link; 
}; 
struct node1 { 
int nst[20]; 
}; 
void insert(int ,char, int); 
int findalpha(char); 
void findfinalstate(void); 
int insertdfastate(struct node1); 
int compare(struct node1,struct node1); 
void printnewstate(struct node1); 
static int set[20],nostate,noalpha,s,notransition,nofinal,start,finalstate[20],c,r,buffer[20]; 
int complete=-1; 
char alphabet[20]; 
static int eclosure[20][20]={0}; 
struct node1 hash[20]; 
struct node * transition[20][20]={NULL}; 
void main() { 
int i,j,k,m,t,n,l; 
struct node *temp; 
struct node1 newstate={0},tmpstate={0}; 
printf("Enter the number of alphabets?\n"); 
printf("NOTE:- [ use letter e as epsilon]\n"); 
printf("NOTE:- [e must be last character ,if it is present]\n"); 
printf("\nEnter No of alphabets and alphabets?\n"); 
scanf("%d",&noalpha); 
getchar(); 
for(i=0;i<noalpha;i++) { 
alphabet[i]=getchar(); 
getchar(); 
} 
printf("Enter the number of states?\n"); 
scanf("%d",&nostate); 
printf("Enter the start state?\n"); 
scanf("%d",&start); 
printf("Enter the number of final states?\n"); 
scanf("%d",&nofinal); 
printf("Enter the final states?\n"); 
for(i=0;i<nofinal;i++) 
scanf("%d",&finalstate[i]); 
printf("Enter no of transition?\n"); 
scanf("%d",&notransition); 
printf("NOTE:- [Transition is in the format[qno alphabet qno]\n",notransition); 
printf("NOTE:- [States number must be greater than zero]\n"); 
printf("\nEnter transition?\n"); 
for(i=0;i<notransition;i++) { 
scanf("%d %c%d",&r,&c,&s); 
insert(r,c,s); 
} 
for(i=0;i<20;i++) { 
for(j=0;j<20;j++) 
hash[i].nst[j]=0; 
} 
complete=-1; 
i=-1; 
printf("\nEquivalent DFA.....\n"); 
printf("Trnsitions of DFA\n"); 
newstate.nst[start]=start; 
insertdfastate(newstate); 
while(i!=complete) { 
i++; 
newstate=hash[i]; 
for(k=0;k<noalpha;k++) { 
c=0; 
for(j=1;j<=nostate;j++) 
set[j]=0; 
for(j=1;j<=nostate;j++) { 
l=newstate.nst[j]; 
if(l!=0) { 
temp=transition[l][k]; 
while(temp!=NULL) { 
if(set[temp->st]==0) { 
c++; 
set[temp->st]=temp->st; 
} 
temp=temp->link; 
} 
} 
} 
printf("\n"); 
if(c!=0) { 
for(m=1;m<=nostate;m++) 
tmpstate.nst[m]=set[m]; 
insertdfastate(tmpstate); 
printnewstate(newstate); 
printf("%c\t",alphabet[k]); 
printnewstate(tmpstate); 
printf("\n"); 
} 
else { 
printnewstate(newstate); 
printf("%c\t", alphabet[k]); 
printf("NULL\n"); 
} 
} 
} 
printf("\nStates of DFA:\n"); 
for(i=0;i<=complete;i++) 
printnewstate(hash[i]);   
printf("\n Alphabets:\n"); 
for(i=0;i<noalpha;i++) 
printf("%c\t",alphabet[i]); 
printf("\n Start State:\n"); 
printf("q%d",start); 
printf("\nFinal states:\n"); 
findfinalstate(); 
} 
int insertdfastate(struct node1 newstate) { 
int i; 
for(i=0;i<=complete;i++) { 
if(compare(hash[i],newstate)) 
return 0; 
} 
complete++; 
hash[complete]=newstate; 
return 1; 
} 
int compare(struct node1 a,struct node1 b) { 
int i; 
for(i=1;i<=nostate;i++) { 
if(a.nst[i]!=b.nst[i]) 
return 0; 
} 
return 1; 
} 
void insert(int r,char c,int s) { 
int j; 
struct node *temp; 
j=findalpha(c); 
if(j==999) { 
printf("error\n"); 
exit(0); 
} 
temp=(struct node *) malloc(sizeof(struct node)); 
temp->st=s; 
temp->link=transition[r][j]; 
transition[r][j]=temp; 
} 
int findalpha(char c) { 
int i; 
for(i=0;i<noalpha;i++) 
if(alphabet[i]==c) 
return i; 
return(999); 
} 
void findfinalstate() { 
int i,j,k,t; 
for(i=0;i<=complete;i++) { 
for(j=1;j<=nostate;j++) { 
for(k=0;k<nofinal;k++) { 
if(hash[i].nst[j]==finalstate[k]) { 
printnewstate(hash[i]); 
printf("\t"); 
j=nostate; 
break; 
} 
} 
} 
} 
} 
void printnewstate(struct node1 state) { 
int j; 
printf("{"); 
for(j=1;j<=nostate;j++) { 
if(state.nst[j]!=0) 
printf("q%d,",state.nst[j]); 
} 
printf("}\t"); } 


-----------------------------------------------------------------------------------------------------------------

output 
Enter the number of alphabets? 
NOTE:- [ use letter e as epsilon] 
NOTE:- [e must be last character ,if it is present] 
Enter No of alphabets and alphabets? 
2 
a 
b 
Enter the number of states? 
3 
Enter the start state? 
1 
Enter the number of final states? 
1 
Enter the final states? 
3 
Enter no of transition? 
4 
NOTE:- [Transition is in the formGÇô> qno alphabet qno] 
NOTE:- [States number must be greater than zero] 
 
Enter transition? 
1 a 1 
1 b 1 
1 a 2 
2 b 3 
 
Equivalent DFA..... 
Trnsitions of DFA 
 
{q1,}   a       {q1,q2,} 
{q1,}   b       {q1,} 
{q1,q2,}        a       {q1,q2,} 
{q1,q2,}        b       {q1,q3,} 
{q1,q3,}        a       {q1,q2,} 
{q1,q3,}        b       {q1,} 
 
States of DFA: 
{q1,}   {q1,q2,}        {q1,q3,} 
 Alphabets: 
a       b 
 Start State: 
q1 
Final states: 
{q1,q3,} --------------
        
-----------------------------------------------------------------------------------------------------------------
        `
    },

{
    id:8,
    name:"Remove left Factoring",
    code:
    `
    #include<stdio.h> 
#include<string.h> 
int main() 
{ 
char gram[20],part1[20],part2[20],modifiedGram[20],newGram[20],tempGram[20]; 
int i,j=0,k=0,l=0,pos; 
printf("Enter Production : A->"); 
gets(gram); 
for(i=0;gram[i]!='|';i++,j++) 
part1[j]=gram[i]; 
part1[j]='\0'; 
for(j=++i,i=0;gram[j]!='\0';j++,i++) 
part2[i]=gram[j]; 
part2[i]='\0'; 
for(i=0;i<strlen(part1)||i<strlen(part2);i++){ 
if(part1[i]==part2[i]){ 
modifiedGram[k]=part1[i]; 
k++; 
pos=i+1; 
} 
} 
for(i=pos,j=0;part1[i]!='\0';i++,j++){ 
newGram[j]=part1[i]; 
} 
newGram[j++]='|'; 
for(i=pos;part2[i]!='\0';i++,j++){ 
newGram[j]=part2[i]; 
} 
modifiedGram[k]='X'; 
modifiedGram[++k]='\0'; 
newGram[j]='\0'; 
printf("\nGrammar Without Left Factoring : : \n"); 
printf(" A->%s",modifiedGram); 
printf("\n X->%s\n",newGram); 
} 

-----------------------------------------------------------------------------------------------------------------

output 
Enter Production : A->aB|aC 
A->aX 
X->B|C 

-----------------------------------------------------------------------------------------------------------------
    `
},

{
    id:9,
    name:"Remove left Recursion",
    code:
    `
    #include<stdio.h> 
#define SIZE 10 
void main ()  
{ 
char non_terminal; 
char beta,alpha; 
char production[SIZE]; 
int index=3;
            /* starting of the string following "->" */ 
printf("Enter the grammar:\n"); 
scanf("%s",production); 
non_terminal=production[0]; 
if(non_terminal==production[index])  
{ 
alpha=production[index+1]; 
printf("Grammar is left recursive.\n"); 
while(production[index]!=0 && production[index]!='|') 
index++; 
if(production[index]!=0)  
{ 
beta=production[index+1]; 
printf("Grammar without left recursion:\n"); 
printf("%c->%c%c'",non_terminal,beta,non_terminal); 
printf("\n%c'->%c%c'|E\n",non_terminal,alpha,non_terminal); 
} 
else 
printf("Grammar can't be reduced\n"); 
} 
else 
printf("Grammar is not left recursive.\n"); 
getch(); 
}

    `
},

{
    id:10,
    name:"Recursive Descent Parser",
    code:
    `
    #include<stdio.h> 
#include<conio.h> 
#include<string.h> 
int i=0,e=0,n=0,l,j; 
char str[50]; 
void E(); 
void F(); 
void E_(); 
void T(); 
void T_(); 
void ed(); 
int main() { 
printf("Recursive descent parser\n\n"); 
printf("E-->TE'\n\tE'-->+TE'/e\n\tT-->FT'\n\tT'-->*FT'/e\n\tF-->(E)/id"); 
printf("\n enter the input exp:"); 
scanf("%s",str); 
l=strlen(str); 
E(); 
if(e>0) 
printf("string is not ACCEPTED"); 
else { 
if(n==0&&e==0&&i==1) 
printf("string is not accepted"); 
else 
printf("string is accepted"); 
} 
getch(); 
} 
void E() { 
T(); 
E_(); 
} 
void T() { 
F(); 
T_(); 
} 
void T_() { 
if(str[i]=='*') { 
  i++; 
  F(); 
  T_(); 
 } 
} 
void E_() { 
 if(str[i]=='+')  { 
  i++; 
  T(); 
  E_(); 
 } 
} 
void F() { 
 if(str[i]=='i'&&str[i+1]=='d') 
 i+=2; 
 else if(str[i]=='(') { 
  i++; 
  n++; 
  E(); 
  ed(); 
 } 
 else e++; 
} 
void ed() { 
 if(str[i]==')') { 
  i++; 
  n--; 
  T_(); 
  E_(); 
 } 
 else e++; 
} 


-----------------------------------------------------------------------------------------------------------------

output: 
Recursive descent parser 
E-->TE' 
        E'-->+TE'/e 
        T-->FT' 
        T'-->*FT'/e 
        F-->(E)/id 
 enter the input exp:id+id*id 
string is accepted ------------------------------ 

-----------------------------------------------------------------------------------------------------------------
    `
},

{
    id:11,
    name:"First and Follow",
    code:
    `
    #include<stdio.h> 
#include<string.h> 
#define max 20 
char prod[max][10]; 
char ter[10],nt[10]; 
char first[10][10],follow[10][10]; 
int eps[10]; 
int count=0; 
int findpos(char ch) { 
int n; 
for(n=0;nt[n]!='\0';n++) 
if(nt[n]==ch) 
break; 
if(nt[n]=='\0') 
return 1; 
return n; 
} 
int IsCap(char c) { 
if(c >= 'A' && c<= 'Z') 
return 1; 
return 0; 
} 
void add(char *arr,char c) { 
int i,flag=0; 
for(i=0;arr[i]!='\0';i++) { 
if(arr[i] == c) { 
flag=1; 
break; 
} 
} 
if(flag!=1) 
arr[strlen(arr)] = c; 
} 
void addarr(char *s1,char *s2) { 
int i,j,flag=99; 
for(i=0;s2[i]!='\0';i++) { 
flag=0; 
for(j=0;;j++) { 
if(s2[i]==s1[j]) { 
flag=1; 
break; 
} 
if(j==strlen(s1) && flag!=1) { 
s1[strlen(s1)] = s2[i]; 
break; 
} 
} 
} 
} 
void addprod(char *s) { 
int i; 
prod[count][0] = s[0]; 
for(i=3;s[i]!='\0';i++) { 
if(!IsCap(s[i])) 
add(ter,s[i]); 
prod[count][i-2] = s[i]; 
} 
prod[count][i-2] = '\0'; 
add(nt,s[0]); 
count++; 
} 
void findfirst() { 
int i,j,n,k,e,n1; 
for(i=0;i<count;i++) { 
for(j=0;j<count;j++) { 
n = findpos(prod[j][0]); 
if(prod[j][1] == (char)238) 
eps[n] = 1; 
else { 
for(k=1,e=1;prod[j][k]!='\0' && e==1;k++) { 
if(!IsCap(prod[j][k])) { 
e=0; 
add(first[n],prod[j][k]); 
} 
else { 
n1 = findpos(prod[j][k]); 
addarr(first[n],first[n1]); 
if(eps[n1] == 0) 
e=0; 
} 
} 
if(e==1) 
eps[n]=1; 
} 
} 
} 
} 
void findfollow() { 
int i,j,k,n,e,n1; 
n = findpos(prod[0][0]); 
add(follow[n],'$'); 
for(i=0;i<count;i++) { 
for(j=0;j<count;j++) { 
k = strlen(prod[j])-1; 
for(;k>0;k--) { 
if(IsCap(prod[j][k])) { 
n=findpos(prod[j][k]); 
if(prod[j][k+1] == '\0') // A -> aB { 
n1 = findpos(prod[j][0]); 
addarr(follow[n],follow[n1]); 
} 
if(IsCap(prod[j][k+1])) // A -> aBb { 
n1 = findpos(prod[j][k+1]); 
addarr(follow[n],first[n1]); 
if(eps[n1]==1) { 
n1=findpos(prod[j][0]); 
addarr(follow[n],follow[n1]); 
} 
} 
else if(prod[j][k+1] != '\0') 
add(follow[n],prod[j][k+1]); 
} 
} 
} 
} 
} 
void main() { 
char s[max],i; 
printf("\nEnter the productions(type 'end' at the last of the production)\n"); 
scanf("%s",s); 
while(strcmp("end",s)) { 
addprod(s); 
scanf("%s",s); 
} 
findfirst(); 
findfollow(); 
printf("NT.....First \t Follow\n"); 
printf("--------------------------\n"); 
for(i=0;i<strlen(nt);i++) { 
printf("%c\t",nt[i]); 
printf("%s",first[i]); 
if(eps[i]==1) 
printf("%c\t",(char)238); 
else 
printf("\t"); 
printf("%s\n",follow[i]); 
} 
printf("--------------------------\n"); 
} 


-----------------------------------------------------------------------------------------------------------------

output 
Enter the productions(type 'end' at the last of the production) 
E->TX 
X->+TX 
T->FY 
Y->*FY 
F->(E) 
F->i 
end 
NT.....First  Follow --------------------- 
E       (i $) 
X        + $) 
T       (i + 
Y        * + 
F       (i * 

-----------------------------------------------------------------------------------------------------------------
    `
}
]
export default ans;